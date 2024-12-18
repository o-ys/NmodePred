import torch as th
import torch.nn.functional as F
import dgl
import numpy as np
from torch import nn
import dgl.function as fn

###the model architecture of graph transformer is modified from https://github.com/BioinfoMachineLearning/DeepInteract
###the model architecture of graph transformer is modified from https://github.com/sc8668/RTMScore

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_input_feats, num_output_feats, num_heads, using_bias=False, update_edge_feats=True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        self.Q = nn.Linear(num_input_feats, num_output_feats * num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, num_output_feats * num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, num_output_feats * num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, num_output_feats * num_heads, bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        for weight in [self.Q.weight, self.K.weight, self.V.weight, self.edge_feats_projection.weight]:
            glorot_orthogonal(weight, scale=scale)
        for bias in [self.Q.bias, self.K.bias, self.V.bias, self.edge_feats_projection.bias]:
            if bias is not None:
                bias.data.fill_(0)

    def propagate_attention(self, g):
        g.apply_edges(lambda edges: {"score": edges.src['K_h'] * edges.dst['Q_h']})
        g.apply_edges(lambda edges: {"score": (edges.data["score"] / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)})
        g.apply_edges(lambda edges: {"score": edges.data['score'] * edges.data['proj_e']})
        if self.update_edge_feats:
            g.apply_edges(lambda edges: {"e_out": edges.data["score"]})

        g.apply_edges(lambda edges: {"score": th.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0, 5.0))})
        g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            e_out = None
            g.ndata['Q_h'] = self.Q(node_feats).view(-1, self.num_heads, self.num_output_feats)
            g.ndata['K_h'] = self.K(node_feats).view(-1, self.num_heads, self.num_output_feats)
            g.ndata['V_h'] = self.V(node_feats).view(-1, self.num_heads, self.num_output_feats)
            g.edata['proj_e'] = self.edge_feats_projection(edge_feats).view(-1, self.num_heads, self.num_output_feats)

            self.propagate_attention(g)

            h_out = g.ndata['wV'] / (g.ndata['z'] + th.full_like(g.ndata['z'], 1e-6))
            if self.update_edge_feats:
                e_out = g.edata['e_out']

            return h_out, e_out

class GraphTransformerModule(nn.Module):
    def __init__(self, num_hidden_channels, activ_fn=nn.SiLU(), residual=True, num_attention_heads=4, norm_to_apply='batch', dropout_rate=0.1):
        super(GraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_output_feats = num_hidden_channels

        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            num_hidden_channels,
            num_hidden_channels // num_attention_heads,
            num_attention_heads,
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.O_edge_feats = nn.Linear(num_hidden_channels, num_hidden_channels)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(num_hidden_channels, num_hidden_channels * 2, bias=False),
            activ_fn,
            dropout,
            nn.Linear(num_hidden_channels * 2, num_hidden_channels, bias=False)
        ])

        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(num_hidden_channels, num_hidden_channels * 2, bias=False),
            activ_fn,
            dropout,
            nn.Linear(num_hidden_channels * 2, num_hidden_channels, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def forward(self, g, node_feats, edge_feats):
        node_feats, edge_feats = self.mha_module(g, node_feats, edge_feats)
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)
        return node_feats, edge_feats

class DGLGraphTransformer(nn.Module):
    def __init__(self, in_channels, edge_features, num_hidden_channels=128, activ_fn=nn.SiLU(), residual=True, num_attention_heads=4, norm_to_apply='batch', dropout_rate=0.1, num_layers=4):
        super(DGLGraphTransformer, self).__init__()

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        num_intermediate_layers = max(0, num_layers - 1)
        self.gt_block = nn.ModuleList([
            GraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate
            ) for _ in range(num_intermediate_layers)
        ])

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.node_encoder(node_feats)
        edge_feats = self.edge_encoder(edge_feats)

        g.edata['h'] = edge_feats
        g.edata['bond'] = edge_feats

        for gt_layer in self.gt_block:
            node_feats, edge_feats = gt_layer(g, node_feats, edge_feats)

        return node_feats

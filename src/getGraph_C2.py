#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import numpy as np
import dgl
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import torch.nn as nn

# Residue-to-embedding dictionary
emb={
    ('ALA', 'BB_COC'): 0, ('ALA', 'SC_COC'): 1,
    ('ARG', 'BB_COC'): 2, ('ARG', 'SC_COC'): 3, 
    ('ASN', 'BB_COC'): 4, ('ASN', 'SC_COC'): 5,
    ('ASP', 'BB_COC'): 6, ('ASP', 'SC_COC'): 7,
    ('CYS', 'BB_COC'): 8, ('CYS', 'SC_COC'): 9,
    ('CYX', 'BB_COC'): 10, ('CYX', 'SC_COC'): 11,
    ('GLN', 'BB_COC'): 12, ('GLN', 'SC_COC'): 13,
    ('GLU', 'BB_COC'): 14, ('GLU', 'SC_COC'): 15,
    ('GLY', 'BB_COC'): 16, ('GLY', 'SC_COC'): 17,
    ('HIE', 'BB_COC'): 18, ('HIE', 'SC_COC'): 19,
    ('ILE', 'BB_COC'): 20, ('ILE', 'SC_COC'): 21,
    ('LEU', 'BB_COC'): 22, ('LEU', 'SC_COC'): 23,
    ('LYS', 'BB_COC'): 24, ('LYS', 'SC_COC'): 25,
    ('MET', 'BB_COC'): 26, ('MET', 'SC_COC'): 27,
    ('PHE', 'BB_COC'): 28, ('PHE', 'SC_COC'): 29,
    ('PRO', 'BB_COC'): 30, ('PRO', 'SC_COC'): 31,
    ('SER', 'BB_COC'): 32, ('SER', 'SC_COC'): 33,
    ('THR', 'BB_COC'): 34, ('THR', 'SC_COC'): 35,
    ('TRP', 'BB_COC'): 36, ('TRP', 'SC_COC'): 37,
    ('TYR', 'BB_COC'): 38, ('TYR', 'SC_COC'): 39,
    ('VAL', 'BB_COC'): 40, ('VAL', 'SC_COC'): 41
}

sc_con={
    'GLY': None,
    'ALA': [("CA", "CB")],
    'CYS': [("CA", "CB"), ("CB", "SG")],
    'CYX': [("CA", "CB"), ("CB", "SG")],
    'CYM': [("CA", "CB"), ("CB", "SG")],
    'SER': [("CA", "CB"), ("CB", "OG")],
    'THR': [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    'VAL': [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
    'PRO': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    'ASP': [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],  
    'ASH': [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],  
    'ASN': [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    'ILE': [("CA", "CB"), ("CB", "CG1"), ("CG1", "CD1"), ("CB", "CG2")],
    'LEU': [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    'MET': [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    'GLU': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    'GLH': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    'GLN': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    'LYS': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    'HIE': [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
    'HID': [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
    'HIP': [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
    'ARG': [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    'PHE': [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"),("CE2", "CD2"),("CD2", "CG")],
    'TYR': [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "OH"), ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG")],
    'TRP': [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"), ("CZ3", "CE3"), ("CE3", "CD2"), ("CD2", "CE2"), ("CG", "CD2")]
}   

# RBF Embedding layer
class RBFEmbedding(nn.Module):
    def __init__(self, num_centers, gamma_init=0.5):
        super(RBFEmbedding, self).__init__()
        self.centers = torch.linspace(0, 8, num_centers)
        self.gamma = torch.tensor(gamma_init)

    def forward(self, distances):
        centers = self.centers.to(distances.device)
        gamma = self.gamma.to(distances.device)
        diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
        rbf = torch.exp(-gamma * diff.pow(2))
        return rbf

# Calculate bond features
def calc_bond_features(bond_type, dist):
    embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=2)
    rbf_emb = RBFEmbedding(num_centers=30)

    bond_type_feat = embedding_layer(torch.tensor([bond_type])).detach().numpy().flatten().tolist()
    rbf_feat = rbf_emb(torch.tensor([dist])).squeeze(0).detach().numpy().tolist()

    return torch.tensor(bond_type_feat + rbf_feat, dtype=torch.float32)

# Compute SASA

def obtainSASA(pdb):
    trajectory = md.load(pdb)
    sasa = md.shrake_rupley(trajectory, mode='atom')
    return {i: sasa[0, i] for i in range(sasa.shape[1])}

# Backbone and sidechain features
def obtainSelfDist(residue):
    try:
        dists = distances.self_distance_array(residue.atoms.positions)
        return [dists.max() * 0.1, dists.min() * 0.1]
    except:
        return [0, 0]

def obtainSCConDist(residue, resname):
    try:
        dists = distances.self_distance_array(residue.atoms.positions)
        max_dist, min_dist = dists.max() * 0.1, dists.min() * 0.1
        return [max_dist, min_dist]
    except:
        return [0, 0]

# Residue bead generation
def get_one_residue_beads(residue):
    bb_atoms = residue.atoms.select_atoms("backbone")
    sc_atoms = residue.atoms.select_atoms("not backbone")

    return {
        "BB_COC": bb_atoms.center_of_geometry() if len(bb_atoms) > 0 else None,
        "SC_COC": sc_atoms.center_of_geometry() if len(sc_atoms) > 0 else None
    }

# Generate protein graph
def getProtGraph(pdb):
    u = mda.Universe(pdb)
    residues = u.residues

    bead_positions, bead_feats = [], []
    sasa_dict = obtainSASA(pdb)

    for residue in residues:
        beads = get_one_residue_beads(residue)
        for bead_type, position in beads.items():
            if position is not None:
                bead_positions.append(position)
                node_key = (residue.resname, bead_type)

                f1 = torch.tensor([emb[node_key]], dtype=torch.long)
                sasa_val = [sasa_dict.get(atom.index, 0) for atom in residue.atoms]
                f2 = [np.mean(sasa_val) if sasa_val else 0]

                bead_feats.append(f1 + f2)

    graph = dgl.graph(([], []), num_nodes=len(bead_positions))
    graph.ndata['h'] = torch.tensor(bead_feats, dtype=torch.float32)

    bead_positions = torch.tensor(bead_positions, dtype=torch.float32)
    dist_matrix = torch.cdist(bead_positions, bead_positions)
    D = torch.where(dist_matrix < 6, dist_matrix, torch.tensor(0.0)).to_sparse()

    src_list, dst_list, bond_feats_all = [], [], []

    for i, j, dist in zip(D.indices()[0], D.indices()[1], D.values()):
        if dist > 0:
            src_list.append(i.item())
            dst_list.append(j.item())

            bond_type = 0  # Set as sequential bond or non-bond based on logic
            bond_feats = calc_bond_features(bond_type, dist)
            bond_feats_all.append(bond_feats)

    graph.add_edges(src_list, dst_list)
    graph.edata['bond'] = torch.stack(bond_feats_all)

    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True, help='Path to the PDB file')
    args = parser.parse_args()

    graph = getProtGraph(args.pdb)
    print(graph)

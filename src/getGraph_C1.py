#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import numpy as np
import dgl
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from rdkit import Chem
import torch.nn as nn

emb={
    ('ALA', 'N'): 0, ('ALA', 'CA'): 1, ('ALA', 'C'): 2, ('ALA', 'O'): 3, ('ALA', 'SC_COC'): 4, ('ALA', 'SC_FA'): 5,
    ('ARG', 'N'): 6, ('ARG', 'CA'): 7, ('ARG', 'C'): 8, ('ARG', 'O'): 9, ('ARG', 'SC_COC'): 10, ('ARG', 'SC_FA'): 11,
    ('ASN', 'N'): 12, ('ASN', 'CA'): 13, ('ASN', 'C'): 14, ('ASN', 'O'): 15, ('ASN', 'SC_COC'): 16, ('ASN', 'SC_FA'): 17,
    ('ASP', 'N'): 18, ('ASP', 'CA'): 19, ('ASP', 'C'): 20, ('ASP', 'O'): 21, ('ASP', 'SC_COC'): 22, ('ASP', 'SC_FA'): 23,
    ('CYS', 'N'): 24, ('CYS', 'CA'): 25, ('CYS', 'C'): 26, ('CYS', 'O'): 27, ('CYS', 'SC_COC'): 28, ('CYS', 'SC_FA'): 29,
    ('CYX', 'N'): 30, ('CYX', 'CA'): 31, ('CYX', 'C'): 32, ('CYX', 'O'): 33, ('CYX', 'SC_COC'): 34, ('CYX', 'SC_FA'): 35,
    ('GLN', 'N'): 36, ('GLN', 'CA'): 37, ('GLN', 'C'): 38, ('GLN', 'O'): 39, ('GLN', 'SC_COC'): 40, ('GLN', 'SC_FA'): 41,
    ('GLU', 'N'): 42, ('GLU', 'CA'): 43, ('GLU', 'C'): 44, ('GLU', 'O'): 45, ('GLU', 'SC_COC'): 46, ('GLU', 'SC_FA'): 47,
    ('GLY', 'N'): 48, ('GLY', 'CA'): 49, ('GLY', 'C'): 50, ('GLY', 'O'): 51, ('GLY', 'SC_COC'): 52, ('GLY', 'SC_FA'): 53,
    ('HIE', 'N'): 54, ('HIE', 'CA'): 55, ('HIE', 'C'): 56, ('HIE', 'O'): 57, ('HIE', 'SC_COC'): 58, ('HIE', 'SC_FA'): 59,
    ('ILE', 'N'): 60, ('ILE', 'CA'): 61, ('ILE', 'C'): 62, ('ILE', 'O'): 63, ('ILE', 'SC_COC'): 64, ('ILE', 'SC_FA'): 65,
    ('LEU', 'N'): 66, ('LEU', 'CA'): 67, ('LEU', 'C'): 68, ('LEU', 'O'): 69, ('LEU', 'SC_COC'): 70, ('LEU', 'SC_FA'): 71,
    ('LYS', 'N'): 72, ('LYS', 'CA'): 73, ('LYS', 'C'): 74, ('LYS', 'O'): 75, ('LYS', 'SC_COC'): 76, ('LYS', 'SC_FA'): 77,
    ('MET', 'N'): 78, ('MET', 'CA'): 79, ('MET', 'C'): 80, ('MET', 'O'): 81, ('MET', 'SC_COC'): 82, ('MET', 'SC_FA'): 83,
    ('PHE', 'N'): 84, ('PHE', 'CA'): 85, ('PHE', 'C'): 86, ('PHE', 'O'): 87, ('PHE', 'SC_COC'): 88, ('PHE', 'SC_FA'): 89,
    ('PRO', 'N'): 90, ('PRO', 'CA'): 91, ('PRO', 'C'): 92, ('PRO', 'O'): 93, ('PRO', 'SC_COC'): 94, ('PRO', 'SC_FA'): 95,
    ('SER', 'N'): 96, ('SER', 'CA'): 97, ('SER', 'C'): 98, ('SER', 'O'): 99, ('SER', 'SC_COC'): 100, ('SER', 'SC_FA'): 101,
    ('THR', 'N'): 102, ('THR', 'CA'): 103, ('THR', 'C'): 104, ('THR', 'O'): 105, ('THR', 'SC_COC'): 106, ('THR', 'SC_FA'): 107,
    ('TRP', 'N'): 108, ('TRP', 'CA'): 109, ('TRP', 'C'): 110, ('TRP', 'O'): 111, ('TRP', 'SC_COC'): 112, ('TRP', 'SC_FA'): 113,
    ('TYR', 'N'): 114, ('TYR', 'CA'): 115, ('TYR', 'C'): 116, ('TYR', 'O'): 117, ('TYR', 'SC_COC'): 118, ('TYR', 'SC_FA'): 119,
    ('VAL', 'N'): 120, ('VAL', 'CA'): 121, ('VAL', 'C'): 122, ('VAL', 'O'): 123, ('VAL', 'SC_COC'): 124, ('VAL', 'SC_FA'): 125
}


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def obtainSASA(pdb):
    trajectory = md.load(pdb)
    sasa = md.shrake_rupley(trajectory, mode='atom')
    sasa_dict = {i: sasa[0, i] for i in range(sasa.shape[1])}
    return sasa_dict

# Define layers for feature transformation
embedding_layer_e1 = nn.Embedding(num_embeddings=2, embedding_dim=32)
linear_layer_e2 = nn.Linear(24, 32)

def rbf_featurizer(dist):
    centers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]
    sigma = 0.15
    transformed_dist = [torch.exp(-0.5 * ((dist - center) ** 2) / sigma**2) for center in centers]
    return transformed_dist

def calc_bond_features(bond_type, dist):
    bond_type_transformed = embedding_layer_e1(torch.tensor([bond_type])).detach().numpy().flatten().tolist()
    scaled_dist = rbf_featurizer(dist)
    scaled_dist_tensor = torch.tensor(scaled_dist, dtype=torch.float32).unsqueeze(0)
    scaled_dist_transformed = linear_layer_e2(scaled_dist_tensor).flatten().tolist()
    return torch.tensor(bond_type_transformed + scaled_dist_transformed, dtype=torch.float32)

def get_one_residue_beads(residue):
    backbone_atoms = residue.atoms.select_atoms("name N or name CA or name C or name O")
    alpha_carbon = residue.atoms.select_atoms("name CA")
    side_chain = residue.atoms.select_atoms("not backbone and not name H*")
    ca_position = alpha_carbon.positions[0] if len(alpha_carbon) > 0 else None
    sc_center = np.mean(side_chain.positions, axis=0) if len(side_chain) > 0 else None
    distances = np.linalg.norm(side_chain.positions - ca_position, axis=1) if len(side_chain) > 0 else np.array([])
    sc_farthest_atom = side_chain[distances.argmax()] if distances.size > 0 else None
    beads = {
        "N": backbone_atoms.select_atoms("name N").positions[0] if len(backbone_atoms.select_atoms("name N")) > 0 else None,
        "CA": ca_position,
        "C": backbone_atoms.select_atoms("name C").positions[0] if len(backbone_atoms.select_atoms("name C")) > 0 else None,
        "O": backbone_atoms.select_atoms("name O").positions[0] if len(backbone_atoms.select_atoms("name O")) > 0 else None,
        "SC_COC": sc_center,
        "SC_FA": sc_farthest_atom.position if sc_farthest_atom is not None else None
    }
    return beads, distances

embedding_layer_f1 = nn.Embedding(num_embeddings=len(emb), embedding_dim=32)
linear_layer_f2 = nn.Linear(1, 32)

def getProtGraph(pdb):
    u = mda.Universe(pdb)
    all_residues_beads = [get_one_residue_beads(residue) for residue in u.residues]
    bead_positions = []
    bead_feats = []
    sasa_dict = obtainSASA(pdb)

    for residue, beads, distances in all_residues_beads:
        for bead_type, position in beads.items():
            if position is not None:
                bead_positions.append(position)
                node_key = (residue.resname, bead_type)
                idx_residue = emb[node_key]
                f1 = embedding_layer_f1(torch.tensor([idx_residue])).detach().numpy().flatten().tolist()
                if bead_type == "SC_COC":
                    sasa_values = [sasa_dict[atom.index] for atom in residue.atoms if not atom.name.startswith("H")]
                    f2 = [np.mean(sasa_values) if sasa_values else 0]
                elif bead_type == "SC_FA":
                    sc_farthest_atom_index = distances.argmax() if distances.size > 0 else None
                    f2 = [sasa_dict[sc_farthest_atom_index] if sc_farthest_atom_index is not None else 0]
                else:
                    atom_index = residue.atoms.select_atoms(f"name {bead_type}")[0].index
                    f2 = [sasa_dict[atom_index]]
                f2 = linear_layer_f2(torch.tensor(f2, dtype=torch.float32)).detach().numpy().flatten().tolist()
                bead_feats.append(f1 + f2)

    bead_positions = np.array(bead_positions)
    bead_feats = np.array(bead_feats)
    graph = dgl.graph(([], []), num_nodes=len(bead_positions))
    graph.ndata['h'] = torch.tensor(bead_feats, dtype=torch.float32)
    bead_positions = torch.tensor(bead_positions, dtype=torch.float32)
    dist_matrix = torch.cdist(bead_positions, bead_positions)
    D = torch.where(dist_matrix < 4, dist_matrix, torch.tensor(0.0)).to_sparse()
    src_list, dst_list, bond_feats_all = [], [], []

    for i, j, dist in zip(D.indices()[0], D.indices()[1], D.values()):
        if dist > 0:
            src_list.append(i.item())
            dst_list.append(j.item())
            bond_type = 0
            bond_feats = calc_bond_features(bond_type, dist)
            bond_feats_all.append(bond_feats)

    graph.add_edges(src_list, dst_list)
    graph.edata['bond'] = torch.stack(bond_feats_all)
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True, help='PDB file path')
    args = parser.parse_args()
    graph = getProtGraph(args.pdb)
    print(graph)

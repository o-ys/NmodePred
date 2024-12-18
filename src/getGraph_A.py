#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import numpy as np
import dgl
from scipy.spatial import distance_matrix
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn as nn

emb_noH={('ALA', 'C'): 0, ('ALA', 'CA'): 1, ('ALA', 'CB'): 2, ('ALA', 'N'): 3, ('ALA', 'O'): 4, ('ARG', 'C'): 5, ('ARG', 'CA'): 6, ('ARG', 'CB'): 7, ('ARG', 'CD'): 8, ('ARG', 'CG'): 9, ('ARG', 'CZ'): 10, ('ARG', 'N'): 11, ('ARG', 'NE'): 12, ('ARG', 'NH1'): 13, ('ARG', 'NH2'): 14, ('ARG', 'O'): 15, ('ASN', 'C'): 16, ('ASN', 'CA'): 17, ('ASN', 'CB'): 18, ('ASN', 'CG'): 19, ('ASN', 'N'): 20, ('ASN', 'ND2'): 21,
         ('ASN', 'O'): 22, ('ASN', 'OD1'): 23, ('ASP', 'C'): 24, ('ASP', 'CA'): 25, ('ASP', 'CB'): 26, ('ASP', 'CG'): 27, ('ASP', 'N'): 28, ('ASP', 'O'): 29, ('ASP', 'OD1'): 30, ('ASP', 'OD2'): 31, ('CYS', 'C'): 32, ('CYS', 'CA'): 33, ('CYS', 'CB'): 34, ('CYS', 'N'): 35, ('CYS', 'O'): 36, ('CYS', 'SG'): 37, ('CYX', 'C'): 38, ('CYX', 'CA'): 39, ('CYX', 'CB'): 40, ('CYX', 'N'): 41, ('CYX', 'O'): 42, ('CYX', 'SG'): 43,
         ('GLN', 'C'): 44, ('GLN', 'CA'): 45, ('GLN', 'CB'): 46, ('GLN', 'CD'): 47, ('GLN', 'CG'): 48, ('GLN', 'N'): 49, ('GLN', 'NE2'): 50, ('GLN', 'O'): 51, ('GLN', 'OE1'): 52, ('GLU', 'C'): 53, ('GLU', 'CA'): 54, ('GLU', 'CB'): 55, ('GLU', 'CD'): 56, ('GLU', 'CG'): 57, ('GLU', 'N'): 58, ('GLU', 'O'): 59, ('GLU', 'OE1'): 60, ('GLU', 'OE2'): 61, ('GLY', 'C'): 62, ('GLY', 'CA'): 63, ('GLY', 'N'): 64, ('GLY', 'O'): 65,
         ('HIE', 'C'): 66, ('HIE', 'CA'): 67, ('HIE', 'CB'): 68, ('HIE', 'CD2'): 69, ('HIE', 'CE1'): 70, ('HIE', 'CG'): 71, ('HIE', 'N'): 72, ('HIE', 'ND1'): 73, ('HIE', 'NE2'): 74, ('HIE', 'O'): 75, ('ILE', 'C'): 76, ('ILE', 'CA'): 77, ('ILE', 'CB'): 78, ('ILE', 'CD1'): 79, ('ILE', 'CG1'): 80, ('ILE', 'CG2'): 81, ('ILE', 'N'): 82, ('ILE', 'O'): 83, ('LEU', 'C'): 84, ('LEU', 'CA'): 85, ('LEU', 'CB'): 86, 
         ('LEU', 'CD1'): 87, ('LEU', 'CD2'): 88, ('LEU', 'CG'): 89, ('LEU', 'N'): 90, ('LEU', 'O'): 91, ('LYS', 'C'): 92, ('LYS', 'CA'): 93, ('LYS', 'CB'): 94, ('LYS', 'CD'): 95, ('LYS', 'CE'): 96, ('LYS', 'CG'): 97, ('LYS', 'N'): 98, ('LYS', 'NZ'): 99, ('LYS', 'O'): 100, ('MET', 'C'): 101, ('MET', 'CA'): 102, ('MET', 'CB'): 103, ('MET', 'CE'): 104, ('MET', 'CG'): 105, ('MET', 'N'): 106, ('MET', 'O'): 107, 
         ('MET', 'SD'): 108, ('PHE', 'C'): 109, ('PHE', 'CA'): 110, ('PHE', 'CB'): 111, ('PHE', 'CD1'): 112, ('PHE', 'CD2'): 113, ('PHE', 'CE1'): 114, ('PHE', 'CE2'): 115, ('PHE', 'CG'): 116, ('PHE', 'CZ'): 117, ('PHE', 'N'): 118, ('PHE', 'O'): 119, ('PRO', 'C'): 120, ('PRO', 'CA'): 121, ('PRO', 'CB'): 122, ('PRO', 'CD'): 123, ('PRO', 'CG'): 124, ('PRO', 'N'): 125, ('PRO', 'O'): 126, ('SER', 'C'): 127, ('SER', 'CA'): 128,
         ('SER', 'CB'): 129, ('SER', 'N'): 130, ('SER', 'O'): 131, ('SER', 'OG'): 132, ('THR', 'C'): 133, ('THR', 'CA'): 134, ('THR', 'CB'): 135, ('THR', 'CG2'): 136, ('THR', 'N'): 137, ('THR', 'O'): 138, ('THR', 'OG1'): 139, ('TRP', 'C'): 140, ('TRP', 'CA'): 141, ('TRP', 'CB'): 142, ('TRP', 'CD1'): 143, ('TRP', 'CD2'): 144, ('TRP', 'CE2'): 145, ('TRP', 'CE3'): 146, ('TRP', 'CG'): 147, ('TRP', 'CH2'): 148, ('TRP', 'CZ2'): 149, 
         ('TRP', 'CZ3'): 150, ('TRP', 'N'): 151, ('TRP', 'NE1'): 152, ('TRP', 'O'): 153, ('TYR', 'C'): 154, ('TYR', 'CA'): 155, ('TYR', 'CB'): 156, ('TYR', 'CD1'): 157, ('TYR', 'CD2'): 158, ('TYR', 'CE1'): 159, ('TYR', 'CE2'): 160, ('TYR', 'CG'): 161, ('TYR', 'CZ'): 162, ('TYR', 'N'): 163, ('TYR', 'O'): 164, ('TYR', 'OH'): 165, ('VAL', 'C'): 166, ('VAL', 'CA'): 167, ('VAL', 'CB'): 168, ('VAL', 'CG1'): 169, ('VAL', 'CG2'): 170, ('VAL', 'N'): 171, ('VAL', 'O'): 172}

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def obtainSASA(pdb):
    import mdtraj as md
    trajectory = md.load(pdb)
    sasa = md.shrake_rupley(trajectory, mode='atom')
    return sasa[0]

# Define embedding and linear layers for feature transformation
embedding_layer_e1 = nn.Embedding(num_embeddings=2, embedding_dim=32)
linear_layer_e2 = nn.Linear(16, 32)

def rbf_featurizer(dist):
    centers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    sigma = 0.15
    dist = torch.tensor(dist, dtype=torch.float32)
    transformed_dist = [torch.exp(-0.5 * ((dist - center) ** 2) / sigma**2) for center in centers]
    return transformed_dist

def calc_bond_features(bond, dist):
    bond_feats = []
    bond_type = 0 if bond is None else 1
    bond_type_transformed = embedding_layer_e1(torch.tensor([bond_type])).detach().numpy().flatten().tolist()
    scaled_dist = rbf_featurizer(dist)
    scaled_dist_tensor = torch.tensor(scaled_dist, dtype=torch.float32).unsqueeze(0)
    scaled_dist_transformed = linear_layer_e2(scaled_dist_tensor).flatten().tolist()
    bond_feats = bond_type_transformed + scaled_dist_transformed
    return torch.tensor(bond_feats, dtype=torch.float32)

embedding_layer_f1 = nn.Embedding(num_embeddings=len(emb_noH), embedding_dim=60)
linear_layer_f2 = nn.Linear(1, 4)

def getProtGraph(pdb):
    u = mda.Universe(pdb)
    all_atoms_noH = u.select_atoms('not name H* and not name OXT')
    num_atoms_noH = len(all_atoms_noH)

    # Create an empty graph
    graph = dgl.graph(([], []), num_nodes=num_atoms_noH)
    sasa_list = obtainSASA(pdb)
    atom_feats = []

    # Guess bonds and filter non-hydrogen bonds
    u.atoms.guess_bonds()
    edge_idx = [(bond.atoms[0].index, bond.atoms[1].index) for bond in u.bonds if not (bond.atoms[0].name.startswith('H') or bond.atoms[1].name.startswith('H'))]

    # Compute distance matrix for edge features
    distance_matrix = distance_array(u.atoms.positions, u.atoms.positions)
    edges_dist = [distance_matrix[i, j] for i, j in edge_idx]

    # Node features
    for atom in all_atoms_noH:
        idx = atom.index
        f1 = embedding_layer_f1(torch.tensor([emb_noH[atom.resname, atom.name]])).detach().numpy().flatten().tolist()
        f2 = torch.tensor([sasa_list[idx]], dtype=torch.float32).unsqueeze(0)
        f2 = linear_layer_f2(f2).detach().numpy().flatten().tolist()
        atom_feats.append(f1 + f2)

    graph.ndata['h'] = torch.tensor(atom_feats, dtype=torch.float32)

    # Use RDKit for bond information
    mol = Chem.MolFromPDBFile(pdb, removeHs=True)
    adj = mol.GetConformer().GetPositions()
    adj = torch.as_tensor(adj)
    dist_matrix = torch.cdist(adj, adj)

    # Identify pairs with distance < 4 Å
    D = torch.where(dist_matrix < 4, dist_matrix, torch.tensor(0.0)).to_sparse()
    D_indices = D.indices()
    D_values = D.values()

    # Add edges and edge features
    src_list, dst_list, bond_feats_all = [], [], []
    for idx in range(D_indices.shape[1]):
        i = D_indices[0, idx].item()
        j = D_indices[1, idx].item()
        dist = D_values[idx].item()
        src_list.append(i)
        dst_list.append(j)
        bond = mol.GetBondBetweenAtoms(i, j)
        bond_feats = calc_bond_features(bond, dist)
        bond_feats_all.append(bond_feats)

    graph.add_edges(src_list, dst_list)
    bond_feats_all_tensor = torch.stack(bond_feats_all)
    graph.edata['bond'] = bond_feats_all_tensor

    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True, help='PDB file path')
    args = parser.parse_args()
    graph = getProtGraph(args.pdb)
    print(graph)

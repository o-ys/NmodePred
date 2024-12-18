import argparse
import pickle
import torch
import dgl
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from Bio.PDB import PDBParser, HSExposure
import torch.nn as nn

# Utility function to load pickled files
def loadPickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Residue types and classifications
allowable_res26 = ['ALA', 'ARG', 'ASN', 'ASP', 'ASH', 'CYS', 'CYX', 'CYM', 'GLN', 'GLU', 'GLH', 'GLY', 'HID', 'HIE', 'HIP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
allowable_sec8 = ['0', '1', '2', '3', '4', '5', '6', '7']
allowable_hyd3 = ['pho', 'phi', 'amphi']
hydrophobic = ['GLY', 'PRO', 'PHE', 'ALA', 'ILE', 'LEU', 'VAL']
hydrophilic = ['SER', 'THR', 'ASP', 'ASH', 'GLU', 'GLH', 'CYS', 'CYX', 'CYM', 'ASN', 'GLN', 'ARG', 'HID', 'HIE', 'HIP']
amphipathic = ['LYS', 'TRP', 'TYR', 'MET']

# One-hot encoding functions
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f'Input {x} not in allowable set {allowable_set}')
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Helper functions for coordinates and distances
def getCAlines(pdb):
    return [line for line in open(pdb) if line.startswith('ATOM') and line.split()[2] == 'CA']

def getXYZ(res_pdbline):
    return [float(res_pdbline.split()[5]), float(res_pdbline.split()[6]), float(res_pdbline.split()[7])]

def getCAxyzList(pdb):
    return np.array([getXYZ(line) for line in getCAlines(pdb)], dtype=np.float32)

def getDistanceBetweenCoords(xyz1, xyz2):
    return np.linalg.norm(np.array(xyz1) - np.array(xyz2))

# Hydropathy classification
def separateHydropathy(res):
    if res.resname in hydrophobic:
        return 'pho'
    elif res.resname in hydrophilic:
        return 'phi'
    elif res.resname in amphipathic:
        return 'amphi'
    else:
        raise ValueError('Invalid residue for hydropathy classification')

# Feature calculation functions
def obtainSASA(pdb):
    import mdtraj as md
    trajectory = md.load(pdb)
    return md.shrake_rupley(trajectory, mode='residue')[0]

def obtainHSE(pdb, idx):
    p = PDBParser(QUIET=True)
    struct = p.get_structure('PDB', pdb)
    hsecb = HSExposure.HSExposureCB(struct)
    res_nums = {i[1][1]: j[:2] for i, j in hsecb.property_dict.items()}
    return res_nums.get(idx, [0, 0])

def obtainSelfDist(res):
    try:
        dists = distances.self_distance_array(res.atoms.positions)
        ca = res.atoms.select_atoms("name CA")
        c = res.atoms.select_atoms("name C")
        n = res.atoms.select_atoms("name N")
        o = res.atoms.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1, distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]

def obtainDihedralAngles(res):
    try:
        return [
            res.phi_selection().dihedral.value() * 0.01 if res.phi_selection() else 0,
            res.psi_selection().dihedral.value() * 0.01 if res.psi_selection() else 0,
            res.omega_selection().dihedral.value() * 0.01 if res.omega_selection() else 0,
            res.chi1_selection().dihedral.value() * 0.01 if res.chi1_selection() else 0
        ]
    except:
        return [0, 0, 0, 0]

# Feature transformations
def rbf_featurizer(dist, cutoff):
    centers = [0.25 * i for i in range(1, int(cutoff * 4) + 1)]
    sigma = 0.15
    dist = torch.tensor(dist, dtype=torch.float32)
    return [torch.exp(-0.5 * ((dist - center) ** 2) / sigma ** 2) for center in centers]

def calc_bond_features(bond_type, dist, cutoff):
    embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=32)
    linear_layer = nn.Linear(len(rbf_featurizer(0, cutoff)), 32)

    bond_type_feat = embedding_layer(torch.tensor([bond_type])).detach().numpy().flatten().tolist()
    rbf_feat = rbf_featurizer(dist, cutoff)
    rbf_tensor = torch.tensor(rbf_feat, dtype=torch.float32).unsqueeze(0)
    scaled_dist_feat = linear_layer(rbf_tensor).detach().numpy().flatten().tolist()

    return torch.tensor(bond_type_feat + scaled_dist_feat, dtype=torch.float32)

# Graph generation
def getProtGraph(pdb, dssp, cutoff):
    u = mda.Universe(pdb)
    ss_list = loadPickle(dssp)
    num_residues = len(u.residues)

    graph = dgl.graph(([], []), num_nodes=num_residues)

    sasa_list = obtainSASA(pdb)
    ca_positions = getCAxyzList(pdb)
    com_positions = np.array([res.atoms.center_of_mass() for res in u.residues], dtype=np.float32)
    res_feats = []

    for res, res_ssnum in zip(u.residues, ss_list):
        idx = res.resindex
        f1 = one_of_k_encoding(res.resname, allowable_res26)
        f2 = one_of_k_encoding(res_ssnum, allowable_sec8)
        f3 = one_of_k_encoding(separateHydropathy(res), allowable_hyd3)
        f4 = [sasa_list[idx]]
        f5 = obtainHSE(pdb, idx)
        f6 = [getDistanceBetweenCoords(ca_positions[idx], com_positions[idx])]
        f7 = obtainDihedralAngles(res)
        f8 = obtainSelfDist(res)

        res_feats.append(torch.tensor(f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8, dtype=torch.float32))

    graph.ndata['h'] = torch.stack(res_feats)

    dist_matrix = distance_array(com_positions, com_positions)
    dist_tensor = torch.tensor(dist_matrix)
    edge_mask = dist_tensor < cutoff

    src, dst = edge_mask.nonzero(as_tuple=True)
    graph.add_edges(src, dst)

    bond_feats = [
        calc_bond_features(1 if abs(i - j) == 1 else 0, dist_matrix[i, j], cutoff)
        for i, j in zip(src.tolist(), dst.tolist())
    ]

    graph.edata['e_ij'] = torch.stack(bond_feats)

    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True, help='Path to PDB file')
    parser.add_argument('--dssp', type=str, required=True, help='Path to DSSP file')
    parser.add_argument('--cutoff', type=float, default=8.0, help='Cutoff distance for edge creation')
    args = parser.parse_args()

    graph = getProtGraph(args.pdb, args.dssp, args.cutoff)
    print(graph)

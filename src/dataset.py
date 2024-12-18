import os
import dgl
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset class for handling structured data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, U, L, G, M, T, S):
        """
        Parameters:
            data_dir (str): Path to the dataset directory
            U (str): Unlimited dataset size (e.g., 205K, 2050K)
            L (str): Level (Atomistic, Coarse-grained, Residue-level)
            G (str): Graph type identifier
            M (str): Mode (e.g., 4REK, Protein-wise, frame-wise)
            T (str): Dataset type (train, validation, test)
            S (str): S entropy specification
        """
        self.data_dir = data_dir
        self.U = U
        self.L = L
        self.G = G
        self.M = M
        self.T = T
        self.S = S
        self.data = self._load_data()
        self.index_to_key = sorted(self.data.keys())

    def _load_data(self):
        """Load data from JSON file"""
        file_path = f'{self.data_dir}/pdb_path{self.U}_{self.L}_g{self.G}_{self.M}_{self.T}.json'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single data point by index"""
        key = self.index_to_key[idx]
        item = self.data[key]
        length = item['length']
        graph_path = item[f'{self.L}_g{self.G}']
        snorm = item[f'Snorm{self.S}']
        return graph_path, snorm, key, length

# Utility function to load a graph from file

def load_graph_from_path(graph_path):
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    try:
        return torch.load(graph_path)
    except EOFError:
        raise EOFError(f"Error reading file: {graph_path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading file {graph_path}: {e}")

# Custom collate function for batching

def my_collate_fn(batch):
    try:
        graph_list = [load_graph_from_path(item[0]) for item in batch]
        label_list = [float(item[1]) for item in batch]
        key_list = [item[2] for item in batch]
        length_list = [item[3] for item in batch]

        graph_batch = dgl.batch(graph_list)
        label_tensor = torch.tensor(label_list, dtype=torch.float32)
        return graph_batch, label_tensor, key_list, length_list
    except Exception as e:
        raise RuntimeError(f"Error during collation: {e}")

if __name__ == '__main__':
    dataset = MyDataset(
        data_dir='path/to/data',
        U='some_U',
        L='some_L',
        G='some_G',
        M='some_M',
        T='train',
        S='some_S'
    )
    print(f"Dataset length: {len(dataset)}")
    graph_path, label, key, length = dataset[0]
    print(f"Graph Path: {graph_path}, Label: {label}, Key: {key}, Length: {length}")

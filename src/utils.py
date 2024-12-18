import os
import re
import glob
import math
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score
)
from torchmetrics import R2Score, PearsonCorrCoef

# Memory Calculation Functions
def calculate_graph_memory(graph):
    """Estimate memory usage of a DGL graph."""
    total_memory = sum(data.numel() * data.element_size() for data in graph.ndata.values())
    total_memory += sum(data.numel() * data.element_size() for data in graph.edata.values())
    return total_memory

def calculate_tensor_memory(tensor):
    """Calculate memory usage of a tensor."""
    return tensor.numel() * tensor.element_size()

def bytes_to_mb(bytes):
    """Convert bytes to megabytes."""
    return bytes / (1024**2)

def bytes_to_gb(bytes):
    """Convert bytes to gigabytes."""
    return bytes / (1024**3)

# Utility Functions
def save_pickle(variable, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(variable, f)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def tenslist_to_list(tenslist):
    return [item.item() if len(item) == 1 else item.tolist() for item in tenslist]

def tensor_to_list(tensor_list):
    return [tensor.tolist() for tensor in tensor_list]

# Training Configuration
def str_to_bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_device(use_gpu, gpu_idx=None):
    if use_gpu:
        device = torch.device(f'cuda:{gpu_idx}' if gpu_idx is not None else 'cuda')
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calibration Function
def calibration(label, pred, bins=10):
    width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - width, bins) + width / 2

    conf_bin, acc_bin, counts = [], [], []
    for threshold in bin_centers:
        bin_idx = np.logical_and(
            threshold - width / 2 < pred,
            pred <= threshold + width / 2
        )
        conf_mean = pred[bin_idx].mean()
        acc_mean = label[bin_idx].mean()
        if not np.isnan(conf_mean):
            conf_bin.append(conf_mean)
            counts.append(len(pred[bin_idx]))
        if not np.isnan(acc_mean):
            acc_bin.append(acc_mean)

    ece = np.sum(np.abs(np.array(conf_bin) - np.array(acc_bin)) * np.array(counts)) / np.sum(counts)
    return np.array(conf_bin), np.array(acc_bin), ece

# Evaluation Functions
def evaluate_classification(y_list, pred_list):
    y_list = torch.cat(y_list).detach().cpu().numpy()
    pred_list = torch.cat(pred_list).detach().cpu().numpy()

    auroc = roc_auc_score(y_list, pred_list)
    _, _, ece = calibration(y_list, pred_list)

    y_list = y_list.astype(int)
    pred_list = np.round(pred_list).astype(int)

    accuracy = accuracy_score(y_list, pred_list)
    precision = precision_score(y_list, pred_list)
    recall = recall_score(y_list, pred_list)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        'accuracy': accuracy,
        'auroc': auroc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ece': ece
    }

def evaluate_regression(y_list, pred_list, mode='my'):
    y_list = torch.cat(y_list).detach().cpu().numpy()
    pred_list = torch.cat(pred_list).detach().cpu().numpy()

    pearson = PearsonCorrCoef()(torch.tensor(pred_list), torch.tensor(y_list))
    r2 = R2Score()(torch.tensor(y_list), torch.tensor(pred_list))

    return {f'{mode}_pcc': pearson.item(), f'{mode}_r2': r2.item()}

# Logging Configuration
def configure_logger(logging, level='info', log_file=None):
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    logging.basicConfig(
        level=levels[level],
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

# Checkpoint Management
def find_latest_checkpoint(result_dir):
    checkpoint_files = glob.glob(f'{result_dir}/checkpoint_epoch_*_loss_*.pth')
    if checkpoint_files:
        return max(checkpoint_files, key=lambda x: int(Path(x).stem.split('_')[2]))
    return None

def get_latest_checkpoint_index(directory, pattern="checkpoint_epoch_(\d+)_loss_\d+\.\d+\.pth"):
    files = os.listdir(directory)
    indices = [int(re.search(pattern, f).group(1)) for f in files if re.search(pattern, f)]
    return max(indices) if indices else -1

def get_epoch_from_checkpoint(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else -1

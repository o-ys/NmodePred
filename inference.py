import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import MyDataset, my_collate_fn
from src.model_RTM7 import DGLGraphTransformer, RTMScore
from src.utils_6 import *

# Configure logger
def configure_logger(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

# Load model and dataset for inference
def load_model_and_data(args):
    # Load dataset
    dataset = MyDataset(
        data_dir=args.data_dir,
        U=args.U, L=args.L, G=args.G, M=args.M, T=args.T, S=args.S, F=args.F
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate_fn, num_workers=args.num_workers
    )

    # Load model
    protmodel = DGLGraphTransformer(
        in_channels=64,
        edge_features=64,
        num_hidden_channels=int(args.hidden_dim),
        activ_fn=torch.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply='batch',
        dropout_rate=0.15,
        num_layers=8
    )
    model = RTMScore(
        prot_model=protmodel,
        in_channels=int(args.hidden_dim),
        hidden_dim=int(args.hidden_dim),
        dropout_rate=0.10
    ).to(args.device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    return model, data_loader

# Perform inference
def run_inference(model, data_loader, device, output_file):
    results = []

    with torch.no_grad():
        for batch in data_loader:
            graph, _, key_list, length_list = batch
            graph = graph.to(device)

            predictions = model(graph).squeeze()
            predictions = predictions.detach().cpu().numpy()

            for key, length, pred in zip(key_list, length_list, predictions):
                results.append({
                    'key': key,
                    'length': length,
                    'prediction': float(pred)
                })

    # Save results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

# Main function for inference
def main(args):
    print("Starting inference...")

    # Configure device
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f'cuda:{args.gpu_idx}')
    else:
        device = torch.device('cpu')
    args.device = device

    # Load model and data
    model, data_loader = load_model_and_data(args)

    # Run inference
    run_inference(model, data_loader, device, args.output_file)

    print("Inference completed.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--output_file', type=str, default='inference_results.json', help='File to save predictions')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='Whether to use GPU for inference')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index to use if GPU is enabled')
    parser.add_argument('--U', type=str, default='205k', help='U parameter for dataset')
    parser.add_argument('--L', type=str, default='A', help='Level parameter for dataset')
    parser.add_argument('--G', type=int, default=1, help='Graph parameter for dataset')
    parser.add_argument('--M', type=str, default='pw', help='Mode parameter for dataset')
    parser.add_argument('--T', type=str, default='test', help='Type of dataset (train, valid, test)')
    parser.add_argument('--S', type=int, default=1, help='S parameter for dataset')
    parser.add_argument('--F', type=int, default=100, help='Frame parameter for dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for the model')

    args = parser.parse_args()

    # Run the main function
    main(args)

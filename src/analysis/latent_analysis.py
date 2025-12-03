"""
Compute latent space representations from trained CVAE model.

This script loads a trained model and computes latent vectors for the test set.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.cvae import CVAE
from utils.paths import DATA_PROCESSED, HAND_OBJECT_DATA_FILE, CHECKPOINTS_DIR


def compute_latent(model, data_loader, device='cuda'):
    """
    Compute latent space representations using the posterior mean.
    
    Args:
        model: Trained CVAE model
        data_loader: DataLoader for the dataset
        device: Device to run on
    
    Returns:
        Latent vectors as numpy array
    """
    model.eval()
    latent_list = []
    
    with torch.no_grad():
        for obj, hand in data_loader:
            obj, hand = obj.to(device), hand.to(device)
            c = model.object_encoder(obj)
            h = model.hand_encoder(hand)
            # Use the posterior mean as the latent representation
            mu_q, _ = model.posterior_net(h, c)
            latent_list.append(mu_q.cpu().numpy())
    
    latent_all = np.concatenate(latent_list, axis=0)
    return latent_all


def main():
    parser = argparse.ArgumentParser(description='Compute latent space representations')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to hand_object_data.pkl')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=64,
                       help='Latent dimension (must match trained model)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save latent vectors')
    
    args = parser.parse_args()
    
    # Determine paths
    model_path = Path(args.model_path) if args.model_path else CHECKPOINTS_DIR / 'best_cvae_hand_pose.pth'
    data_file = Path(args.data_file) if args.data_file else HAND_OBJECT_DATA_FILE
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the requested split
    hand_data = data[f'hand_{args.split}']
    obj_data = data[f'obj_{args.split}']
    obj_names = np.array(data['obj_names'])[data[f'{args.split}_indices']]
    
    # Create data loader
    dataset = TensorDataset(
        torch.tensor(obj_data, dtype=torch.float32),
        torch.tensor(hand_data, dtype=torch.float32)
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    model = CVAE(latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Compute latent vectors
    print(f"Computing latent vectors for {args.split} set...")
    latent_vectors = compute_latent(model, data_loader, device=device)
    
    print(f"Computed latent vectors shape: {latent_vectors.shape}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent.parent / 'outputs' / f'latent_{args.split}.pkl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'latent_vectors': latent_vectors,
        'obj_names': obj_names,
        'split': args.split
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Latent vectors saved to {output_path}")
    print(f"Object names shape: {obj_names.shape}")
    print(f"Unique objects: {len(np.unique(obj_names))}")


if __name__ == '__main__':
    main()


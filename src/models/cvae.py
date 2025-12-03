"""
Conditional Variational Autoencoder (CVAE) for hand-object interaction modeling.

This module contains the CVAE model architecture and training code.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_PROCESSED, HAND_OBJECT_DATA_FILE, CHECKPOINTS_DIR, METRICS_DIR


class ObjectEncoder(nn.Module):
    """Encoder for object features."""
    def __init__(self, input_dim=7869, hidden_dims=[512, 256], out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], out_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class HandEncoder(nn.Module):
    """Encoder for hand features."""
    def __init__(self, input_dim=114, hidden_dim=128, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class PriorNet(nn.Module):
    """Prior network for latent distribution."""
    def __init__(self, cond_dim=128, latent_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(cond_dim, latent_dim)
    
    def forward(self, c):
        return self.fc_mu(c), self.fc_logvar(c)


class PosteriorNet(nn.Module):
    """Posterior network for latent distribution."""
    def __init__(self, cond_dim=128, hand_dim=64, latent_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(cond_dim + hand_dim, latent_dim)
        self.fc_logvar = nn.Linear(cond_dim + hand_dim, latent_dim)
    
    def forward(self, h, c):
        x = torch.cat([h, c], dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Decoder for hand reconstruction."""
    def __init__(self, latent_dim=64, cond_dim=128, hidden_dims=[256, 512], output_dim=114):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], output_dim)
        )
    
    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1)
        return self.fc(x)


class CVAE(nn.Module):
    """Conditional Variational Autoencoder for hand-object interaction."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.object_encoder = ObjectEncoder()
        self.hand_encoder = HandEncoder()
        self.prior_net = PriorNet(cond_dim=128, latent_dim=latent_dim)
        self.posterior_net = PosteriorNet(cond_dim=128, hand_dim=64, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obj, hand):
        c = self.object_encoder(obj)
        h = self.hand_encoder(hand)
        mu_q, logvar_q = self.posterior_net(h, c)
        mu_p, logvar_p = self.prior_net(c)
        z = self.reparameterize(mu_q, logvar_q)
        recon_hand = self.decoder(z, c)
        return recon_hand, mu_q, logvar_q, mu_p, logvar_p


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    """Compute KL divergence between posterior and prior."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * torch.sum(logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)**2) / var_p, dim=1)
    return torch.mean(kl)


def get_beta(epoch, kl_anneal_epochs=20):
    """KL annealing schedule."""
    return min(1.0, epoch / kl_anneal_epochs)


def train(model, train_loader, val_loader, epochs=50, device='cuda', 
          checkpoint_dir=None, metrics_dir=None):
    """
    Train the CVAE model.
    
    Args:
        model: CVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        metrics_dir: Directory to save metrics
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    recon_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    
    # Metric tracking
    train_losses = []
    val_losses = []
    val_recon_losses = []
    val_pose_errors = []
    val_joints_errors = []
    val_trans_errors = []
    
    # Hand vector dimensions
    hand_pose_dim = 48
    joints_dim = 63
    trans_dim = 3
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0
        beta = get_beta(epoch, kl_anneal_epochs=20)
        
        for obj, hand in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            obj, hand = obj.to(device), hand.to(device)
            optimizer.zero_grad()
            recon_hand, mu_q, logvar_q, mu_p, logvar_p = model(obj, hand)
            recon_loss = recon_loss_fn(recon_hand, hand)
            kl_loss = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0
        val_recon_loss_epoch = 0
        pose_error_epoch = 0
        joints_error_epoch = 0
        trans_error_epoch = 0
        
        with torch.no_grad():
            for obj, hand in val_loader:
                obj, hand = obj.to(device), hand.to(device)
                recon_hand, mu_q, logvar_q, mu_p, logvar_p = model(obj, hand)
                recon_loss = recon_loss_fn(recon_hand, hand)
                kl_loss = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
                loss = recon_loss + beta * kl_loss
                val_loss_epoch += loss.item()
                val_recon_loss_epoch += recon_loss.item()
                
                # Compute detailed errors
                pose_err = nn.functional.mse_loss(
                    recon_hand[:, :hand_pose_dim],
                    hand[:, :hand_pose_dim],
                    reduction='sum'
                )
                joints_err = nn.functional.mse_loss(
                    recon_hand[:, hand_pose_dim:hand_pose_dim+joints_dim],
                    hand[:, hand_pose_dim:hand_pose_dim+joints_dim],
                    reduction='sum'
                )
                trans_err = nn.functional.mse_loss(
                    recon_hand[:, hand_pose_dim+joints_dim:],
                    hand[:, hand_pose_dim+joints_dim:],
                    reduction='sum'
                )
                pose_error_epoch += pose_err.item()
                joints_error_epoch += joints_err.item()
                trans_error_epoch += trans_err.item()
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_recon_loss = val_recon_loss_epoch / len(val_loader)
        avg_pose_error = pose_error_epoch / len(val_loader)
        avg_joints_error = joints_error_epoch / len(val_loader)
        avg_trans_error = trans_error_epoch / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_pose_errors.append(avg_pose_error)
        val_joints_errors.append(avg_joints_error)
        val_trans_errors.append(avg_trans_error)
        
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}")
        print(f"Pose Error: {avg_pose_error:.4f}, Joints Error: {avg_joints_error:.4f}, "
              f"Trans Error: {avg_trans_error:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / 'best_cvae_hand_pose.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    
    # Save final metrics
    if metrics_dir:
        losses_errors = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_recon_losses': val_recon_losses,
            'val_pose_errors': val_pose_errors,
            'val_joints_errors': val_joints_errors,
            'val_trans_errors': val_trans_errors
        }
        metrics_path = metrics_dir / 'cvae_losses_errors.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(losses_errors, f)
        print(f"Metrics saved to {metrics_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to hand_object_data.pkl')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--latent-dim', type=int, default=64,
                       help='Latent dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--metrics-dir', type=str, default=None,
                       help='Directory to save metrics')
    
    args = parser.parse_args()
    
    # Determine paths
    data_file = Path(args.data_file) if args.data_file else HAND_OBJECT_DATA_FILE
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINTS_DIR
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else METRICS_DIR
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    hand_train = data['hand_train']
    hand_val = data['hand_val']
    hand_test = data['hand_test']
    obj_train = data['obj_train']
    obj_val = data['obj_val']
    obj_test = data['obj_test']
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(obj_train, dtype=torch.float32),
        torch.tensor(hand_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(obj_val, dtype=torch.float32),
        torch.tensor(hand_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = CVAE(latent_dim=args.latent_dim)
    
    # Train
    print("Starting training...")
    train(model, train_loader, val_loader, epochs=args.epochs, device=device,
          checkpoint_dir=checkpoint_dir, metrics_dir=metrics_dir)
    
    print("Training completed!")


if __name__ == '__main__':
    main()


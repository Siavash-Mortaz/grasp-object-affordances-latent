"""
Plot training and validation metrics from saved training history.

This script loads saved training metrics and creates visualization plots
for training/validation losses and reconstruction errors.
"""
import argparse
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import METRICS_DIR, FIGURES_DIR


def plot_training_metrics(metrics_file, save_dir=None, show=True):
    """
    Plot training and validation metrics.
    
    Args:
        metrics_file: Path to saved metrics pickle file
        save_dir: Directory to save plots (optional)
        show: Whether to display plots
    """
    # Load metrics
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    train_losses = metrics.get('train_losses', [])
    val_losses = metrics.get('val_losses', [])
    val_recon_losses = metrics.get('val_recon_losses', [])
    val_pose_errors = metrics.get('val_pose_errors', [])
    val_joints_errors = metrics.get('val_joints_errors', [])
    val_trans_errors = metrics.get('val_trans_errors', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Train Loss
    axes[0, 0].plot(train_losses, label='Train Loss (MSE in mm²)')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (mm²)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Plot 2: Validation Loss
    axes[0, 1].plot(val_losses, label='Val Loss', color='orange')
    axes[0, 1].set_title('Validation Loss (MSE in mm²)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (mm²)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Plot 3: Validation Reconstruction Loss
    axes[0, 2].plot(val_recon_losses, label='Recon Loss', color='green')
    axes[0, 2].set_title('Validation Reconstruction Loss (MSE in mm²)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss (mm²)')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
    # Plot 4: Pose Error
    if val_pose_errors:
        axes[1, 0].plot(val_pose_errors, label='Pose Error', color='red')
        axes[1, 0].set_title('Validation Pose Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # Plot 5: Joints Error
    if val_joints_errors:
        axes[1, 1].plot(val_joints_errors, label='Joints Error', color='purple')
        axes[1, 1].set_title('Validation Joints Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
    # Plot 6: Translation Error
    if val_trans_errors:
        axes[1, 2].plot(val_trans_errors, label='Trans Error', color='brown')
        axes[1, 2].set_title('Validation Translation Error')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Error')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--metrics-file', type=str, default=None,
                       help='Path to metrics pickle file')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    # Determine metrics file path
    if args.metrics_file:
        metrics_file = Path(args.metrics_file)
    else:
        metrics_file = METRICS_DIR / 'cvae_losses_errors.pkl'
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    # Determine save directory
    save_dir = Path(args.save_dir) if args.save_dir else FIGURES_DIR
    
    plot_training_metrics(metrics_file, save_dir=save_dir, show=not args.no_show)


if __name__ == '__main__':
    main()


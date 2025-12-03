"""
Principal Component Analysis (PCA) of latent space.

This script performs PCA on latent vectors and creates visualizations
grouped by object names, as used in the paper.
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import FIGURES_DIR


def plot_pca_explained_variance(pca, save_dir=None, show=True):
    """Plot PCA explained variance ratio."""
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot cumulative explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cumulative variance
    ax1.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance")
    ax1.set_title("PCA Explained Variance Ratio")
    ax1.grid(True)
    
    # Individual components (first 10 or all if less)
    n_components = min(10, len(explained_variance))
    ax2.bar(range(1, n_components + 1), explained_variance[:n_components],
            tick_label=[f'PC{i}' for i in range(1, n_components + 1)])
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title(f"Explained Variance of First {n_components} Principal Components")
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'pca_explained_variance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print statistics
    print("\nCumulative Explained Variance:")
    for i in range(min(10, len(cumulative_variance))):
        print(f"First {i+1} components explain {cumulative_variance[i]:.2%} of the variance")
    
    print("\nIndividual Component Variance:")
    for i in range(min(10, len(explained_variance))):
        print(f"Principal Component {i+1} explains {explained_variance[i]:.2%} of the variance")


def plot_2d_pca(pca_results, object_names, pc_x=1, pc_y=2, save_dir=None, show=True):
    """Plot 2D PCA scatter plot grouped by object names."""
    pca_df = pd.DataFrame(
        pca_results, 
        columns=[f'PC{i+1}' for i in range(pca_results.shape[1])]
    )
    pca_df['Object Name'] = object_names
    
    unique_objects = np.unique(object_names)
    
    plt.figure(figsize=(12, 8))
    for obj_name in unique_objects:
        subset = pca_df[pca_df['Object Name'] == obj_name]
        plt.scatter(subset[f'PC{pc_x}'], subset[f'PC{pc_y}'], 
                   label=obj_name, alpha=0.6, s=20)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
    plt.xlabel(f"Principal Component {pc_x}")
    plt.ylabel(f"Principal Component {pc_y}")
    plt.title("PCA of Latent Space Grouped by Object Names")
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'pca_2d_pc{pc_x}_pc{pc_y}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_pca(pca_results, object_names, pc_x=1, pc_y=2, pc_z=3, save_dir=None):
    """Plot 3D PCA scatter plot grouped by object names using Plotly."""
    pca_df = pd.DataFrame(
        pca_results,
        columns=[f'PC{i+1}' for i in range(pca_results.shape[1])]
    )
    pca_df['Object Name'] = object_names
    
    fig = px.scatter_3d(
        pca_df,
        x=f'PC{pc_x}',
        y=f'PC{pc_y}',
        z=f'PC{pc_z}',
        color='Object Name',
        hover_name='Object Name',
        title="3D PCA of Latent Space Grouped by Object Names"
    )
    
    fig.update_layout(
        height=1000,
        scene=dict(
            xaxis_title=f'Principal Component {pc_x}',
            yaxis_title=f'Principal Component {pc_y}',
            zaxis_title=f'Principal Component {pc_z}'
        )
    )
    
    if save_dir:
        save_path = Path(save_dir) / f'pca_3d_pc{pc_x}_pc{pc_y}_pc{pc_z}.html'
        fig.write_html(str(save_path))
        print(f"Interactive plot saved to {save_path}")
    
    fig.show()


def plot_3d_pca_with_centroids(pca_results, object_names, pca, pc_x=1, pc_y=2, pc_z=3, save_dir=None):
    """Plot 3D PCA with centroids for each object group."""
    pca_df = pd.DataFrame(
        pca_results,
        columns=[f'PC{i+1}' for i in range(pca_results.shape[1])]
    )
    pca_df['Object Name'] = object_names
    
    # Calculate centroids
    centroid_df = pca_df.groupby("Object Name")[[f'PC{pc_x}', f'PC{pc_y}', f'PC{pc_z}']].mean().reset_index()
    
    # Project centroids back to latent space (for potential use)
    latent_dim = pca_results.shape[1]
    centroids_full_pca = np.zeros((centroid_df.shape[0], latent_dim))
    centroids_full_pca[:, [pc_x-1, pc_y-1, pc_z-1]] = centroid_df[[f'PC{pc_x}', f'PC{pc_y}', f'PC{pc_z}']].values
    latent_centroids = pca.inverse_transform(centroids_full_pca)
    
    print("\nCentroids of each cluster:")
    print(centroid_df)
    
    # Create plot
    fig = px.scatter_3d(
        pca_df,
        x=f'PC{pc_x}',
        y=f'PC{pc_y}',
        z=f'PC{pc_z}',
        color='Object Name',
        hover_name='Object Name',
        title="3D PCA of Latent Space Grouped by Object Names"
    )
    
    # Add centroids
    fig.add_trace(
        go.Scatter3d(
            x=centroid_df[f'PC{pc_x}'],
            y=centroid_df[f'PC{pc_y}'],
            z=centroid_df[f'PC{pc_z}'],
            mode='markers+text',
            marker=dict(symbol='x', size=10, color='black'),
            text=centroid_df['Object Name'],
            name="Centroids"
        )
    )
    
    fig.update_layout(
        height=1000,
        scene=dict(
            xaxis_title=f'Principal Component {pc_x}',
            yaxis_title=f'Principal Component {pc_y}',
            zaxis_title=f'Principal Component {pc_z}'
        )
    )
    
    if save_dir:
        save_path = Path(save_dir) / f'pca_3d_centroids_pc{pc_x}_pc{pc_y}_pc{pc_z}.html'
        fig.write_html(str(save_path))
        print(f"Interactive plot saved to {save_path}")
    
    fig.show()
    
    return centroid_df, latent_centroids


def main():
    parser = argparse.ArgumentParser(description='Perform PCA analysis on latent space')
    parser.add_argument('--latent-file', type=str, default=None,
                       help='Path to latent vectors pickle file')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (default: all)')
    parser.add_argument('--pc-x', type=int, default=7,
                       help='X-axis principal component (default: 7)')
    parser.add_argument('--pc-y', type=int, default=8,
                       help='Y-axis principal component (default: 8)')
    parser.add_argument('--pc-z', type=int, default=9,
                       help='Z-axis principal component (default: 9)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--with-centroids', action='store_true',
                       help='Include centroids in 3D plot')
    
    args = parser.parse_args()
    
    # Load latent vectors
    if args.latent_file:
        latent_file = Path(args.latent_file)
    else:
        # Try to find default location
        latent_file = Path(__file__).parent.parent.parent / 'outputs' / 'latent_test.pkl'
    
    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")
    
    print(f"Loading latent vectors from {latent_file}...")
    with open(latent_file, 'rb') as f:
        data = pickle.load(f)
    
    latent_vectors = data['latent_vectors']
    object_names = data['obj_names']
    
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Object names shape: {object_names.shape}")
    
    # Perform PCA
    n_components = args.n_components if args.n_components else latent_vectors.shape[1]
    print(f"\nPerforming PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(latent_vectors)
    
    print(f"PCA results shape: {pca_results.shape}")
    
    # Determine save directory
    save_dir = Path(args.save_dir) if args.save_dir else FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot explained variance
    plot_pca_explained_variance(pca, save_dir=save_dir, show=not args.no_show)
    
    # Plot 2D PCA
    plot_2d_pca(pca_results, object_names, 
                pc_x=args.pc_x, pc_y=args.pc_y,
                save_dir=save_dir, show=not args.no_show)
    
    # Plot 3D PCA
    if args.with_centroids:
        plot_3d_pca_with_centroids(pca_results, object_names, pca,
                                   pc_x=args.pc_x, pc_y=args.pc_y, pc_z=args.pc_z,
                                   save_dir=save_dir)
    else:
        plot_3d_pca(pca_results, object_names,
                   pc_x=args.pc_x, pc_y=args.pc_y, pc_z=args.pc_z,
                   save_dir=save_dir)
    
    # Save PCA results
    pca_output = save_dir / 'pca_results.pkl'
    with open(pca_output, 'wb') as f:
        pickle.dump({
            'pca_results': pca_results,
            'pca_model': pca,
            'object_names': object_names
        }, f)
    print(f"\nPCA results saved to {pca_output}")


if __name__ == '__main__':
    main()


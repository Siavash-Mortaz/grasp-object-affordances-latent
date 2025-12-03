"""
t-SNE analysis of latent space.

This script performs t-SNE dimensionality reduction on latent vectors
and creates visualizations grouped by object names.
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import FIGURES_DIR


def plot_2d_tsne(tsne_results, object_names, save_dir=None, show=True):
    """Plot 2D t-SNE scatter plot grouped by object names."""
    plt.figure(figsize=(12, 8))
    for obj_name in np.unique(object_names):
        indices = object_names == obj_name
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                   label=obj_name, alpha=0.6, s=20)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE of Latent Space by Object Names")
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'tsne_2d.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_tsne(tsne_results, object_names, save_dir=None):
    """Plot 3D t-SNE scatter plot using Plotly."""
    df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        'TSNE-3': tsne_results[:, 2],
        'Object': object_names
    })
    
    fig = px.scatter_3d(
        df,
        x='TSNE-1',
        y='TSNE-2',
        z='TSNE-3',
        color='Object',
        opacity=1,
        title="3D t-SNE of Latent Space by Object Names"
    )
    
    fig.update_layout(legend_title_text='Object Names')
    
    if save_dir:
        save_path = Path(save_dir) / 'tsne_3d.html'
        fig.write_html(str(save_path))
        print(f"Interactive plot saved to {save_path}")
    
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Perform t-SNE analysis on latent space')
    parser.add_argument('--latent-file', type=str, default=None,
                       help='Path to latent vectors pickle file')
    parser.add_argument('--n-components', type=int, default=2,
                       choices=[2, 3],
                       help='Number of t-SNE components (2 or 3)')
    parser.add_argument('--perplexity', type=float, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    # Load latent vectors
    if args.latent_file:
        latent_file = Path(args.latent_file)
    else:
        latent_file = Path(__file__).parent.parent.parent / 'outputs' / 'latent_test.pkl'
    
    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")
    
    print(f"Loading latent vectors from {latent_file}...")
    with open(latent_file, 'rb') as f:
        data = pickle.load(f)
    
    latent_vectors = data['latent_vectors']
    object_names = data['obj_names']
    
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Performing t-SNE with {args.n_components} components, perplexity={args.perplexity}...")
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        random_state=42,
        learning_rate='auto',
        init='random'
    )
    tsne_results = tsne.fit_transform(latent_vectors)
    
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    # Determine save directory
    save_dir = Path(args.save_dir) if args.save_dir else FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    if args.n_components == 2:
        plot_2d_tsne(tsne_results, object_names, save_dir=save_dir, show=not args.no_show)
    else:
        plot_3d_tsne(tsne_results, object_names, save_dir=save_dir)
    
    # Save t-SNE results
    tsne_output = save_dir / f'tsne_{args.n_components}d_results.pkl'
    with open(tsne_output, 'wb') as f:
        pickle.dump({
            'tsne_results': tsne_results,
            'object_names': object_names,
            'perplexity': args.perplexity
        }, f)
    print(f"\nt-SNE results saved to {tsne_output}")


if __name__ == '__main__':
    main()


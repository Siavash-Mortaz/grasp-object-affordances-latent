"""
K-Means clustering analysis on latent space (PCA or t-SNE results).

This script performs clustering analysis and visualizes results with
elbow method, silhouette scores, and cluster visualizations.
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import FIGURES_DIR


def elbow_method(data, max_k=25, save_dir=None, show=True):
    """Perform elbow method to find optimal number of clusters."""
    inertia = []
    
    print("Computing elbow method...")
    for k in tqdm(range(1, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    if save_dir:
        save_path = Path(save_dir) / 'elbow_method.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return inertia


def silhouette_analysis(data, max_k=25, save_dir=None, show=True):
    """Perform silhouette analysis to find optimal number of clusters."""
    sil_scores = []
    
    print("Computing silhouette scores...")
    for k in tqdm(range(2, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        sil_scores.append(score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), sil_scores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    if save_dir:
        save_path = Path(save_dir) / 'silhouette_scores.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return sil_scores


def plot_kmeans_clusters(data, object_names, n_clusters, save_dir=None, show=True):
    """Plot K-Means clustering results."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    print(f"\nK-Means clustering with k={n_clusters}")
    print(f"Centroids shape: {centroids.shape}")
    
    # For 2D data
    if data.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=20, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='x', 
                   linewidths=3, label='Centroids')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'K-Means Clustering with k = {n_clusters}')
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'kmeans_k{n_clusters}_2d.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # For 3D data
    elif data.shape[1] >= 3:
        df = pd.DataFrame(data[:, :3], columns=['Dim1', 'Dim2', 'Dim3'])
        df['Cluster'] = clusters
        df['Object'] = object_names
        
        fig = px.scatter_3d(
            df,
            x='Dim1',
            y='Dim2',
            z='Dim3',
            color='Cluster',
            title=f'K-Means Clusters with k = {n_clusters}',
            labels={'Cluster': 'Cluster'},
            opacity=1
        )
        
        # Add centroids
        fig.add_scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x', line=dict(color='black', width=2)),
            name='Centroids'
        )
        
        fig.update_layout(height=1200, coloraxis_showscale=False)
        
        if save_dir:
            save_path = Path(save_dir) / f'kmeans_k{n_clusters}_3d.html'
            fig.write_html(str(save_path))
            print(f"Interactive plot saved to {save_path}")
        
        fig.show()
    
    return clusters, centroids


def plot_kmeans_by_objects(data, object_names, n_clusters, save_dir=None, show=True):
    """Plot K-Means clusters with object name labels for centroids."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    # Create DataFrame
    df_data = pd.DataFrame(data[:, :3], columns=['Dim1', 'Dim2', 'Dim3'])
    df_data['Object Name'] = object_names
    df_data['Cluster'] = clusters
    df_data['Type'] = 'Data'
    
    # Create centroids DataFrame
    df_centroids = pd.DataFrame(centroids[:, :3], columns=['Dim1', 'Dim2', 'Dim3'])
    df_centroids['Cluster'] = range(n_clusters)
    df_centroids['Type'] = 'Centroid'
    
    # Find nearest object name for each centroid
    centroid_labels = []
    for i in range(n_clusters):
        cluster_data = df_data[df_data['Cluster'] == i]
        if not cluster_data.empty:
            points = cluster_data[['Dim1', 'Dim2', 'Dim3']].values
            centroid_point = centroids[i, :3].reshape(1, -1)
            distances = cdist(points, centroid_point, metric='euclidean')
            min_index = np.argmin(distances)
            nearest_name = cluster_data.iloc[min_index]['Object Name']
            print(f'Centroid {i+1}, Object Name={nearest_name}')
        else:
            nearest_name = ""
        centroid_labels.append(nearest_name)
    
    df_centroids['Object Name'] = centroid_labels
    
    # Combine dataframes
    df_combined = pd.concat([df_data, df_centroids], ignore_index=True)
    df_combined['Label'] = df_combined.apply(
        lambda row: row['Object Name'] if row['Type'] == 'Centroid' else "", axis=1
    )
    df_combined['Size'] = df_combined['Type'].apply(lambda x: 15 if x == 'Centroid' else 5)
    
    # Create plot
    fig = px.scatter_3d(
        df_combined,
        x='Dim1',
        y='Dim2',
        z='Dim3',
        color='Cluster',
        symbol='Type',
        size='Size',
        text='Label',
        title=f"3D PCA of Latent Space with K-Means Clusters and Centroids (k={n_clusters})",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        )
    )
    
    if save_dir:
        save_path = Path(save_dir) / f'kmeans_k{n_clusters}_with_labels.html'
        fig.write_html(str(save_path))
        print(f"Interactive plot saved to {save_path}")
    
    fig.show()
    
    return clusters, centroids


def main():
    parser = argparse.ArgumentParser(description='Perform K-Means clustering analysis')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to data file (PCA or t-SNE results pickle)')
    parser.add_argument('--n-clusters', type=int, default=7,
                       help='Number of clusters for K-Means')
    parser.add_argument('--max-k', type=int, default=25,
                       help='Maximum k for elbow/silhouette analysis')
    parser.add_argument('--do-elbow', action='store_true',
                       help='Perform elbow method analysis')
    parser.add_argument('--do-silhouette', action='store_true',
                       help='Perform silhouette analysis')
    parser.add_argument('--with-labels', action='store_true',
                       help='Include object name labels for centroids')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    # Load data
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        # Try to find PCA or t-SNE results
        data_file = Path(__file__).parent.parent.parent / 'outputs' / 'figures' / 'pca_results.pkl'
        if not data_file.exists():
            data_file = Path(__file__).parent.parent.parent / 'outputs' / 'latent_test.pkl'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data and object names
    if 'pca_results' in data:
        analysis_data = data['pca_results']
        object_names = data['object_names']
        data_type = 'PCA'
    elif 'tsne_results' in data:
        analysis_data = data['tsne_results']
        object_names = data['object_names']
        data_type = 't-SNE'
    elif 'latent_vectors' in data:
        analysis_data = data['latent_vectors']
        object_names = data['obj_names']
        data_type = 'Latent'
    else:
        raise ValueError("Unknown data format in pickle file")
    
    print(f"Data type: {data_type}")
    print(f"Data shape: {analysis_data.shape}")
    
    # Determine save directory
    save_dir = Path(args.save_dir) if args.save_dir else FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform analyses
    if args.do_elbow:
        elbow_method(analysis_data, max_k=args.max_k, save_dir=save_dir, show=not args.no_show)
    
    if args.do_silhouette:
        silhouette_analysis(analysis_data, max_k=args.max_k, save_dir=save_dir, show=not args.no_show)
    
    # Perform clustering
    if args.with_labels:
        clusters, centroids = plot_kmeans_by_objects(
            analysis_data, object_names, args.n_clusters, save_dir=save_dir, show=not args.no_show
        )
    else:
        clusters, centroids = plot_kmeans_clusters(
            analysis_data, object_names, args.n_clusters, save_dir=save_dir, show=not args.no_show
        )
    
    # Save clustering results
    clustering_output = save_dir / f'clustering_k{args.n_clusters}.pkl'
    with open(clustering_output, 'wb') as f:
        pickle.dump({
            'clusters': clusters,
            'centroids': centroids,
            'n_clusters': args.n_clusters,
            'object_names': object_names
        }, f)
    print(f"\nClustering results saved to {clustering_output}")


if __name__ == '__main__':
    main()


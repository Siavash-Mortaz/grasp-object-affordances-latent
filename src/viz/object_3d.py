"""
Visualize 3D transformed object point clouds.

This script loads object data and visualizes transformed point clouds using
matplotlib and plotly.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_INTERIM, HAND_POSES_FILE, OBJECT_INFOS_FILE, FILE_INFOS_FILE
from data.inspect_extracted import load_saved_data


def apply_transformations(points, obj_trans, obj_rot_matrix):
    """
    Apply rotation and translation to points.
    
    Args:
        points: Point cloud array (N, 3)
        obj_trans: Translation vector (3,)
        obj_rot_matrix: Rotation matrix (3, 3)
    
    Returns:
        Transformed points
    """
    # Apply rotation
    rotated_points = np.dot(points, obj_rot_matrix.T)
    # Apply translation
    translated_points = rotated_points + obj_trans
    return translated_points


def visualize_object_3d(object_infos, index, use_plotly=True):
    """
    Visualize a 3D object point cloud.
    
    Args:
        object_infos: List of object info dictionaries
        index: Index of the object to visualize
        use_plotly: If True, use plotly for interactive visualization
    """
    if index >= len(object_infos):
        raise IndexError(f"Index {index} out of range (max: {len(object_infos)-1})")
    
    obj_trans = object_infos[index]['objTrans']
    obj_rot = object_infos[index]['objRot']
    obj_name = object_infos[index]['objName']
    obj_point = object_infos[index]['objPointCloud']
    
    print(f'Translation shape: {obj_trans.shape}')
    print(f'Rotation: {obj_rot}')
    print(f'Point cloud shape: {obj_point.shape}')
    print(f'Object name: {obj_name}')
    
    # Apply transformations
    rotation = R.from_rotvec(obj_rot.flatten()).as_matrix()
    transformed_points = apply_transformations(obj_point, obj_trans, rotation)
    
    # Extract coordinates
    x = transformed_points[:, 0]
    y = transformed_points[:, 1]
    z = transformed_points[:, 2]
    
    if use_plotly:
        # Create interactive 3D plot with plotly
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            title=f'3D Scatter Plot of Transformed Object Points\nObject File Name: ({obj_name})'
        )
        fig.update_traces(marker=dict(color='blue', size=5))
        fig.update_layout(
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate',
                aspectmode='cube'
            )
        )
        fig.show()
    else:
        # Create static 3D plot with matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'3D Scatter Plot of Transformed Object Points\nObject File Name: ({obj_name})')
        ax.set_box_aspect([1, 1, 1])
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D object point clouds')
    parser.add_argument('--index', type=int, default=0,
                       help='Index of object to visualize')
    parser.add_argument('--in-dir', type=str, default=None,
                       help='Directory containing extracted data files')
    parser.add_argument('--use-plotly', action='store_true', default=True,
                       help='Use plotly for interactive visualization (default: True)')
    parser.add_argument('--use-matplotlib', action='store_true',
                       help='Use matplotlib for static visualization')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.in_dir:
        in_dir = Path(args.in_dir)
        hand_poses_file = in_dir / 'hand_poses.pkl'
        object_infos_file = in_dir / 'object_infos.pkl'
        file_infos_file = in_dir / 'file_infos.pkl'
    else:
        hand_poses_file = HAND_POSES_FILE
        object_infos_file = OBJECT_INFOS_FILE
        file_infos_file = FILE_INFOS_FILE
    
    # Load data
    print("Loading data...")
    hand_poses, object_infos, file_infos = load_saved_data(
        hand_poses_file, object_infos_file, file_infos_file
    )
    
    # Visualize
    use_plotly = args.use_plotly and not args.use_matplotlib
    visualize_object_3d(object_infos, args.index, use_plotly=use_plotly)


if __name__ == '__main__':
    main()


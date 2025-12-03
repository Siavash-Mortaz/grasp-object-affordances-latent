"""
Extract data from HO3D dataset annotations.

This script reads sequence information from text files, extracts relevant annotations
from corresponding pickle files, and saves the processed data into new pickle files.
"""
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_INTERIM, HO3D_ROOT, HO3D_MODELS


def load_xyz(file_path):
    """Load point cloud from .xyz file."""
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                x, y, z = map(float, line.split())
                points.append([x, y, z])
    return np.array(points)


def load_ho3d_best_info(data_dir, models_dir, split='train'):
    """
    Load hand and object information from HO3D dataset.
    
    Args:
        data_dir: Directory where the HO3D dataset is stored
        models_dir: Directory containing object model files (<objName>/points.xyz)
        split: Dataset split to load ('train', 'test', etc.)
    
    Returns:
        hand_poses: List of hand pose dictionaries
        object_infos: List of object info dictionaries
        file_infos: List of file info dictionaries
    """
    hand_poses = []
    object_infos = []
    file_infos = []

    split_file = os.path.join(data_dir, f'{split}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        sequences = f.readlines()

    # Initialize the progress bar
    total_sequences = len(sequences)
    with tqdm(total=total_sequences, desc="Loading data", unit="sequence") as pbar:
        for sequence in sequences:
            seq_name, file_id = sequence.strip().split('/')
            meta_file = os.path.join(data_dir, split, seq_name, 'meta', f'{file_id}.pkl')

            if not os.path.exists(meta_file):
                pbar.update(1)
                continue

            # Load annotations
            with open(meta_file, 'rb') as mf:
                annotations = pickle.load(mf)

            if annotations['handPose'] is not None and annotations['objTrans'] is not None:
                obj_name = annotations['objName']
                xyz_file_path = os.path.join(models_dir, obj_name, 'points.xyz')
                
                if not os.path.exists(xyz_file_path):
                    print(f"Warning: Model file not found: {xyz_file_path}")
                    pbar.update(1)
                    continue
                
                hand_poses.append({
                    'handPose': annotations['handPose'],
                    'handTrans': annotations['handTrans'],
                    'handJoints3D': annotations['handJoints3D']
                })

                object_infos.append({
                    'objTrans': annotations['objTrans'],
                    'objRot': annotations['objRot'],
                    'objName': annotations['objName'],
                    'objLabel': annotations['objLabel'],
                    'objPointCloud': load_xyz(xyz_file_path)
                })

                file_infos.append({
                    'folder': seq_name,
                    'frame': file_id
                })

            # Update the progress bar
            pbar.update(1)

    return hand_poses, object_infos, file_infos


def main():
    parser = argparse.ArgumentParser(description='Extract data from HO3D dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (train, test, etc.)')
    parser.add_argument('--ho3d-root', type=str, default=None, help='Path to HO3D dataset root')
    parser.add_argument('--models-dir', type=str, default=None, help='Path to HO3D models directory')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for extracted data')
    
    args = parser.parse_args()
    
    # Use provided paths or defaults from paths.py
    data_dir = Path(args.ho3d_root) if args.ho3d_root else HO3D_ROOT
    models_dir = Path(args.models_dir) if args.models_dir else HO3D_MODELS
    out_dir = Path(args.out_dir) if args.out_dir else DATA_INTERIM
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    print(f"Using models from: {models_dir}")
    print(f"Saving to: {out_dir}")
    
    # Load data
    hand_poses, object_infos, file_infos = load_ho3d_best_info(
        str(data_dir), str(models_dir), split=args.split
    )
    
    # Save extracted data
    hand_poses_file = out_dir / 'hand_poses.pkl'
    object_infos_file = out_dir / 'object_infos.pkl'
    file_infos_file = out_dir / 'file_infos.pkl'
    
    with open(hand_poses_file, 'wb') as hp_file:
        pickle.dump(hand_poses, hp_file)
    
    with open(object_infos_file, 'wb') as oi_file:
        pickle.dump(object_infos, oi_file)
    
    with open(file_infos_file, 'wb') as fi_file:
        pickle.dump(file_infos, fi_file)
    
    print("\nData saved successfully!")
    print(f"Number of hand poses loaded: {len(hand_poses)}")
    print(f"Number of object infos loaded: {len(object_infos)}")
    print(f"Number of file infos loaded: {len(file_infos)}")


if __name__ == '__main__':
    main()

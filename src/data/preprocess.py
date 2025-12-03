"""
Preprocess extracted data: normalize features and split into train/val/test sets.

This script extracts, normalizes, and combines necessary features, then splits
the data into training, validation, and testing sets.
"""
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_INTERIM, DATA_PROCESSED, HAND_POSES_FILE, OBJECT_INFOS_FILE, FILE_INFOS_FILE, HAND_OBJECT_DATA_FILE, SCALERS_FILE
from data.inspect_extracted import load_saved_data


def preprocess_data(hand_poses, object_infos, file_infos):
    """
    Preprocess hand and object data: extract, normalize, and combine features.
    
    Args:
        hand_poses: List of hand pose dictionaries
        object_infos: List of object info dictionaries
        file_infos: List of file info dictionaries
    
    Returns:
        hand_data, obj_data, obj_names, scalers_dict, folder_names, frame_numbers
    """
    # Extract hand pose information
    hand_pose_data = np.array([pose['handPose'] for pose in hand_poses])
    hand_trans_data = np.array([pose['handTrans'] for pose in hand_poses])
    hand_joints_data = np.array([pose['handJoints3D'] for pose in hand_poses])

    # Extract object information
    obj_trans_data = np.array([info['objTrans'] for info in object_infos])
    obj_rot_data = np.array([info['objRot'] for info in object_infos])
    obj_points_data = np.array([info['objPointCloud'] for info in object_infos])

    # Reshape objRot if necessary
    num_objRot_reshaped = 0
    for i in range(len(obj_rot_data)):
        if obj_rot_data[i].shape != (3,):
            obj_rot_data[i] = obj_rot_data[i].reshape(3)
            num_objRot_reshaped += 1
    if num_objRot_reshaped > 0:
        print(f'Number of reshaped objRot: {num_objRot_reshaped}')

    # Extract object names and file info
    obj_names = [info['objName'] for info in object_infos]
    folder_names = [info['folder'] for info in file_infos]
    frame_numbers = [info['frame'] for info in file_infos]

    # Normalize data using StandardScaler
    # Hand joints: reshape to (N, 63) for scaling, then back to (N, 21, 3)
    scaler_hand_joints = StandardScaler()
    hand_joints_data = scaler_hand_joints.fit_transform(
        hand_joints_data.reshape(-1, 63)
    ).reshape(-1, 21, 3)

    # Object points: reshape to (N, 7863) for scaling, then back to (N, 2621, 3)
    scaler_obj_points = StandardScaler()
    obj_points_data = scaler_obj_points.fit_transform(
        obj_points_data.reshape(-1, 7863)
    ).reshape(-1, 2621, 3)

    # Normalize other parameters
    scaler_hand_pose = StandardScaler()
    hand_pose_data = scaler_hand_pose.fit_transform(hand_pose_data)

    scaler_hand_trans = StandardScaler()
    hand_trans_data = scaler_hand_trans.fit_transform(hand_trans_data)

    scaler_obj_trans = StandardScaler()
    obj_trans_data = scaler_obj_trans.fit_transform(obj_trans_data)

    scaler_obj_rot = StandardScaler()
    obj_rot_data = scaler_obj_rot.fit_transform(obj_rot_data)

    # Combine data
    # Hand: pose (48) + translation (3) + joints (63) = 114 dims
    hand_data = np.hstack((
        hand_pose_data, 
        hand_trans_data, 
        hand_joints_data.reshape(-1, 63)
    ))
    
    # Object: translation (3) + rotation (3) + points (7863) = 7869 dims
    obj_data = np.hstack((
        obj_trans_data, 
        obj_rot_data, 
        obj_points_data.reshape(-1, 7863)
    ))

    scalers = {
        'scaler_hand_joints': scaler_hand_joints,
        'scaler_obj_points': scaler_obj_points,
        'scaler_hand_pose': scaler_hand_pose,
        'scaler_hand_trans': scaler_hand_trans,
        'scaler_obj_trans': scaler_obj_trans,
        'scaler_obj_rot': scaler_obj_rot
    }

    return hand_data, obj_data, obj_names, scalers, folder_names, frame_numbers


def main():
    parser = argparse.ArgumentParser(description='Preprocess extracted data')
    parser.add_argument('--in-dir', type=str, default=None,
                       help='Directory containing extracted data files')
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Output directory for processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.25,
                       help='Proportion of remaining data for validation (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Determine input/output paths
    if args.in_dir:
        in_dir = Path(args.in_dir)
        hand_poses_file = in_dir / 'hand_poses.pkl'
        object_infos_file = in_dir / 'object_infos.pkl'
        file_infos_file = in_dir / 'file_infos.pkl'
    else:
        hand_poses_file = HAND_POSES_FILE
        object_infos_file = OBJECT_INFOS_FILE
        file_infos_file = FILE_INFOS_FILE
    
    out_dir = Path(args.out_dir) if args.out_dir else DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load extracted data
    print("Loading extracted data...")
    hand_poses, object_infos, file_infos = load_saved_data(
        hand_poses_file, object_infos_file, file_infos_file
    )
    
    # Preprocess data
    print("Preprocessing data...")
    hand_data, obj_data, obj_names, scalers, folder_names, frame_numbers = preprocess_data(
        hand_poses, object_infos, file_infos
    )
    
    # Split data into train/val/test
    print(f"Splitting data (total samples: {len(hand_data)})...")
    print(f"Test size: {args.test_size}, Val size: {args.val_size} of remaining data")
    
    indices = np.arange(len(hand_data))
    hand_train, hand_test, obj_train, obj_test, idx_train, idx_test = train_test_split(
        hand_data, obj_data, indices, test_size=args.test_size, random_state=args.seed
    )
    hand_train, hand_val, obj_train, obj_val, idx_train_final, idx_val = train_test_split(
        hand_train, obj_train, idx_train, test_size=args.val_size, random_state=args.seed
    )
    
    print("Training data shape:", hand_train.shape, obj_train.shape)
    print("Validation data shape:", hand_val.shape, obj_val.shape)
    print("Test data shape:", hand_test.shape, obj_test.shape)
    
    # Save processed data
    data_files = {
        'hand_train': hand_train,
        'hand_val': hand_val,
        'hand_test': hand_test,
        'obj_train': obj_train,
        'obj_val': obj_val,
        'obj_test': obj_test,
        'obj_names': obj_names,
        'folder_names': folder_names,
        'frame_numbers': frame_numbers,
        'train_indices': idx_train_final,
        'val_indices': idx_val,
        'test_indices': idx_test
    }
    
    data_file_path = out_dir / 'hand_object_data.pkl'
    scalers_file_path = out_dir / 'scalers.pkl'
    
    with open(data_file_path, 'wb') as data_file:
        pickle.dump(data_files, data_file)
    
    with open(scalers_file_path, 'wb') as scalers_file:
        pickle.dump(scalers, scalers_file)
    
    print(f"\nData and scalers saved successfully!")
    print(f"Data saved to: {data_file_path}")
    print(f"Scalers saved to: {scalers_file_path}")


if __name__ == '__main__':
    main()


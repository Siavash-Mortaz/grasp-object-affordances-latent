"""
Inspect extracted data from pickle files.

This script loads and displays sample information from the extracted data files.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_INTERIM, HAND_POSES_FILE, OBJECT_INFOS_FILE, FILE_INFOS_FILE


def load_saved_data(hand_poses_file, object_infos_file, file_infos_file):
    """
    Load and deserialize data from pickle files.
    
    Args:
        hand_poses_file: Path to hand_poses.pkl
        object_infos_file: Path to object_infos.pkl
        file_infos_file: Path to file_infos.pkl
    
    Returns:
        hand_poses, object_infos, file_infos
    """
    with open(hand_poses_file, 'rb') as hp_file:
        hand_poses = pickle.load(hp_file)

    with open(object_infos_file, 'rb') as oi_file:
        object_infos = pickle.load(oi_file)

    with open(file_infos_file, 'rb') as fi_file:
        file_infos = pickle.load(fi_file)
    
    return hand_poses, object_infos, file_infos


def main():
    parser = argparse.ArgumentParser(description='Inspect extracted data')
    parser.add_argument('--in-dir', type=str, default=None, 
                       help='Directory containing extracted data files')
    parser.add_argument('--hand-poses', type=str, default=None,
                       help='Path to hand_poses.pkl')
    parser.add_argument('--object-infos', type=str, default=None,
                       help='Path to object_infos.pkl')
    parser.add_argument('--file-infos', type=str, default=None,
                       help='Path to file_infos.pkl')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.in_dir:
        in_dir = Path(args.in_dir)
        hand_poses_file = in_dir / 'hand_poses.pkl'
        object_infos_file = in_dir / 'object_infos.pkl'
        file_infos_file = in_dir / 'file_infos.pkl'
    else:
        hand_poses_file = Path(args.hand_poses) if args.hand_poses else HAND_POSES_FILE
        object_infos_file = Path(args.object_infos) if args.object_infos else OBJECT_INFOS_FILE
        file_infos_file = Path(args.file_infos) if args.file_infos else FILE_INFOS_FILE
    
    # Load data
    hand_poses, object_infos, file_infos = load_saved_data(
        hand_poses_file, object_infos_file, file_infos_file
    )
    
    # Print statistics
    print("=" * 50)
    print("Data loaded successfully!")
    print(f"Number of hand poses loaded: {len(hand_poses)}")
    print(f"Number of object infos loaded: {len(object_infos)}")
    print(f"Number of file infos loaded: {len(file_infos)}")
    print("=" * 50)
    
    # Print sample information
    if len(object_infos) > 0:
        print('\n-- Sample Object Info (1st entry) --')
        print(f"Object Name: {object_infos[0]['objName']}")
        print(f"Object Label: {object_infos[0]['objLabel']}")
        print(f"Object Rotation: {object_infos[0]['objRot']}")
        print(f"Object Translation: {object_infos[0]['objTrans']}")
        print(f"Object Point Cloud Shape: {object_infos[0]['objPointCloud'].shape}")
    
    if len(hand_poses) > 0:
        print('\n-- Sample Hand Info (1st entry) --')
        print(f"Hand Pose Shape: {hand_poses[0]['handPose'].shape}")
        print(f"Hand Translation: {hand_poses[0]['handTrans']}")
        print(f"Hand Joints 3D Shape: {hand_poses[0]['handJoints3D'].shape}")
    
    if len(file_infos) > 0:
        print('\n-- Sample File Info (1st entry) --')
        print(f"Folder Name: {file_infos[0]['folder']}")
        print(f"Frame: {file_infos[0]['frame']}")


if __name__ == '__main__':
    main()


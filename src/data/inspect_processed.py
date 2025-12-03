"""
Inspect preprocessed data from pickle files.

This script loads and displays sample information from the preprocessed data files.
"""
import argparse
import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import DATA_PROCESSED, HAND_OBJECT_DATA_FILE, SCALERS_FILE


def main():
    parser = argparse.ArgumentParser(description='Inspect preprocessed data')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to hand_object_data.pkl')
    parser.add_argument('--scalers-file', type=str, default=None,
                       help='Path to scalers.pkl')
    
    args = parser.parse_args()
    
    # Determine file paths
    data_file = Path(args.data_file) if args.data_file else HAND_OBJECT_DATA_FILE
    scalers_file = Path(args.scalers_file) if args.scalers_file else SCALERS_FILE
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open(data_file, 'rb') as f:
        data_files = pickle.load(f)
    
    hand_train = data_files['hand_train']
    hand_val = data_files['hand_val']
    hand_test = data_files['hand_test']
    obj_train = data_files['obj_train']
    obj_val = data_files['obj_val']
    obj_test = data_files['obj_test']
    obj_names = data_files['obj_names']
    folder_names = data_files['folder_names']
    frame_numbers = data_files['frame_numbers']
    idx_train = data_files['train_indices']
    idx_val = data_files['val_indices']
    idx_test = data_files['test_indices']
    
    # Load scalers
    with open(scalers_file, 'rb') as f:
        scalers = pickle.load(f)
    
    print("Data and scalers loaded successfully!")
    print(f"Training data shape: {hand_train.shape}, {obj_train.shape}")
    print(f"Validation data shape: {hand_val.shape}, {obj_val.shape}")
    print(f"Test data shape: {hand_test.shape}, {obj_test.shape}")
    print('=' * 50)
    
    # Print sample information
    print('\n----- Sample Preprocessed Information -----')
    print(f"\n1st Object Information From Training Part:")
    print(f"  Shape: {obj_train[0].shape}")
    print(f"  First 5 values: {obj_train[0][:5]}")
    
    print(f"\n1st Hand Information From Training Part:")
    print(f"  Shape: {hand_train[0].shape}")
    print(f"  First 5 values: {hand_train[0][:5]}")
    
    print(f"\n1st Object Name From Training Part: {obj_names[idx_train[0]]}")
    print(f"1st Folder Information From Training Part: {folder_names[idx_train[0]]}")
    print(f"1st Frame Information From Training Part: {frame_numbers[idx_train[0]]}")
    
    print(f"\n1st Object Information From Test Part:")
    print(f"  Shape: {obj_test[0].shape}")
    print(f"  First 5 values: {obj_test[0][:5]}")
    
    print(f"\n1st Hand Information From Test Part:")
    print(f"  Shape: {hand_test[0].shape}")
    print(f"  First 5 values: {hand_test[0][:5]}")
    
    print(f"\n1st Object Name From Test Part: {obj_names[idx_test[0]]}")
    print(f"1st Folder Information From Test Part: {folder_names[idx_test[0]]}")
    print(f"1st Frame Information From Test Part: {frame_numbers[idx_test[0]]}")
    
    # Print unique objects in test set
    print(f"\nUnique objects in test set:")
    seen = set()
    for i in range(len(obj_test)):
        if obj_names[idx_test[i]] not in seen:
            print(f"  {obj_names[idx_test[i]]}")
            seen.add(obj_names[idx_test[i]])


if __name__ == '__main__':
    main()


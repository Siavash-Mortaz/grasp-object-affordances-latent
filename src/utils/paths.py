"""
Centralized path configuration for the project.
Update these paths according to your local setup.
"""
from pathlib import Path

# Project root directory (assuming this file is in src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# HO3D dataset paths (UPDATE THESE TO YOUR LOCAL PATHS)
# Default paths - update these in your local setup
HO3D_ROOT = Path(r"D:/datasets/HO3D_v3")  # Update this to your HO3D dataset location
HO3D_MODELS = HO3D_ROOT / "models"  # Directory containing <objName>/points.xyz files

# Project data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Create directories if they don't exist
for dir_path in [DATA_INTERIM, DATA_PROCESSED, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file names
HAND_POSES_FILE = DATA_INTERIM / "hand_poses.pkl"
OBJECT_INFOS_FILE = DATA_INTERIM / "object_infos.pkl"
FILE_INFOS_FILE = DATA_INTERIM / "file_infos.pkl"
HAND_OBJECT_DATA_FILE = DATA_PROCESSED / "hand_object_data.pkl"
SCALERS_FILE = DATA_PROCESSED / "scalers.pkl"


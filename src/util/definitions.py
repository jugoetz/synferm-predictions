import pathlib

PROJECT_DIR = pathlib.Path(__file__).parents[2]
DATA_ROOT = PROJECT_DIR / "data"
CONFIG_ROOT = PROJECT_DIR / "config"
LOG_DIR = PROJECT_DIR / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"
TRAINED_MODEL_DIR = PROJECT_DIR / "production_models"
LOG_DIR_ABS = str(LOG_DIR.absolute())

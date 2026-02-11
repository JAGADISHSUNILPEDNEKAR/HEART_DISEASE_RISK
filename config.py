from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# File paths
DATA_FILE = DATA_DIR / "heart_disease_uci.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200

# Feature definitions
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "fbs",
    "exang",
    "ca",
]
CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "slope", "thal"]
TARGET_COLUMN = "num"

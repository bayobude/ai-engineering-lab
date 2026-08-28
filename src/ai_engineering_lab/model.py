from pathlib import Path

import joblib
from sklearn.datasets import load_iris


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the trained model
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load Iris species names
iris = load_iris()
from pathlib import Path

import joblib
from sklearn.datasets import load_iris

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Find the saved model
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load Iris data to get the species names
iris = load_iris()

# Example flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Make prediction
prediction = model.predict(new_flower)

# Convert the predicted number into a species name
predicted_species = iris.target_names[prediction[0]]

print("Predicted species:", predicted_species)
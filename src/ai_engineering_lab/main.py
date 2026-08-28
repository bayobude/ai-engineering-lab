from fastapi import FastAPI

from ai_engineering_lab.model import iris, model
from ai_engineering_lab.schemas import FlowerMeasurements


app = FastAPI(
    title="Iris Classification API",
    description="An API that predicts Iris flower species.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Iris Classification API is running!"
    }


@app.post("/predict")
def predict(flower: FlowerMeasurements):
    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    prediction = model.predict(features)

    predicted_species = iris.target_names[prediction[0]]

    return {
        "prediction": predicted_species
    }
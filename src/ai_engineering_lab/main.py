from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Iris Classification API",
    description="A machine learning API for predicting Iris flower species.",
    version="1.0.0",
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

MODEL_PATH = Path("models/iris_model.joblib")

model = joblib.load(MODEL_PATH)

CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


class PredictionResponse(BaseModel):
    prediction: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    features: list[str]
    classes: list[str]


@app.get("/")
def read_root():
    return {"message": "Iris Classification API is running"}


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health_check():
    return {"status": "healthy"}


@app.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info():
    return {
        "model_name": "Iris Classifier",
        "model_type": type(model).__name__,
        "features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
        "classes": [
            "setosa",
            "versicolor",
            "virginica",
        ],
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(iris: IrisFeatures):
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width,
    ]]

    prediction = model.predict(features)[0]

    class_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }

    return {
        "prediction": class_names[int(prediction)]
    }

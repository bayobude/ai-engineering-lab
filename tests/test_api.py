from fastapi.testclient import TestClient

from ai_engineering_lab.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
         "message": "Iris Classification API is running"
 }


def test_predict_setosa():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "setosa"


def test_invalid_measurement():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -5,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}



def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "Iris Classifier"
    assert "model_type" in data

    assert data["features"] == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    assert data["classes"] == [
        "setosa",
        "versicolor",
        "virginica",
    ]

# Iris Classification API

An end-to-end machine learning project that trains an Iris flower classification model and serves predictions through a FastAPI REST API.

## Features

- Machine learning classification using Scikit-learn
- Logistic Regression model
- Iris dataset
- Model saved using Joblib
- FastAPI REST API
- `/predict` endpoint
- Pydantic input validation
- Automated tests with Pytest
- Dependency management using uv

## Project Structure

```text
ai-engineering-lab/
├── models/
│   └── iris_model.joblib
├── src/
│   └── ai_engineering_lab/
│       ├── __init__.py
│       ├── main.py
│       ├── model.py
│       ├── predict.py
│       ├── schemas.py
│       └── test_setup.py
├── tests/
│   └── test_api.py
├── README.md
├── pyproject.toml
└── uv.lock



Machine Learning Workflow

The project follows this workflow:

Dataset
   ↓
Data Exploration
   ↓
Data Preparation
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model
   ↓
FastAPI Deployment
   ↓
Automated Testing



Installation

Clone the repository:

git clone https://github.com/bayobude/ai-engineering-lab.git

Move into the project directory:

cd ai-engineering-lab

Install the project dependencies:

uv sync
Run the API

Start the FastAPI server:

uv run uvicorn ai_engineering_lab.main:app --app-dir src --reload

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs
API Usage
Predict an Iris Flower Species

Endpoint:

POST /predict

Example request:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Example response:

{
  "prediction": "setosa"
}
Input Validation

The API validates flower measurements using Pydantic.

All measurements must be greater than zero.

Example invalid request:

{
  "sepal_length": -5,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

The API will return a validation error.

Running Tests

Run the automated tests with:

uv run pytest

Expected result:

3 passed

The tests check:

The API homepage
A successful Iris prediction
Invalid measurement validation
Technologies Used
Python
Scikit-learn
NumPy
Pandas
Joblib
FastAPI
Pydantic
Pytest
Uvicorn
uv
Git and GitHub

Author

Adebayo Bude

Future Improvements

Possible future improvements include:

Docker containerization
CI/CD with GitHub Actions
Additional machine learning models
Model comparison and evaluation
API logging
Cloud deployment
Database integration
Monitoring and model versioning

## Step 2: Save the file

Press:

```text
Ctrl + S
Step 3: Check the changes

In your terminal, run:

git status

Then paste the output here.

# AI Engineering Lab — Agent Customizations

## Project Overview
This is an AI Engineering Lab project for data science experiments, machine learning prototyping, and computational exploration. The stack includes:
- **Python**: 3.12+ with numpy, pandas, scikit-learn, matplotlib
- **Execution**: Jupyter notebooks + Python scripts
- **Build**: `uv` build system with `uv.lock`

### Commands
- **Run test**: `python src/ai_engineering_lab/test_setup.py`
- **Install dependencies**: `uv sync`
- **Launch Jupyter**: `jupyter notebook`

---

## JSON Handling Conventions

Since this lab works with data, configs, and experiment metadata, follow these JSON patterns:

### 1. **Experiment Configuration**
Store experiment parameters in JSON:
```json
{
  "experiment_id": "exp_001",
  "model": "random_forest",
  "params": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "dataset": "iris",
  "date": "2026-08-27"
}
```

Load and use:
```python
import json
with open("experiment_config.json") as f:
    config = json.load(f)
```

### 2. **Serializing Results**
Save experiment results as JSON for reproducibility:
```python
import json
from datetime import datetime

results = {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94,
    "timestamp": datetime.now().isoformat(),
    "metrics": {"auc": 0.97}
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 3. **API Data Handling**
When working with APIs (common in AI labs), parse JSON responses:
```python
import json
import requests

response = requests.get("https://api.example.com/data")
data = response.json()  # Directly parse JSON from response

# Validate and normalize
if isinstance(data, dict) and "results" in data:
    results = json.loads(json.dumps(data["results"]))  # Deep copy
```

### 4. **Data Interchange**
Use JSON as a bridge between Python and external tools:
- Export train/test splits metadata
- Store feature engineering pipelines
- Log hyperparameter searches
- Version control-friendly format

### 5. **File Organization**
- **Config**: `configs/*.json` (experiment settings, hyperparameters)
- **Results**: `results/*.json` (metrics, predictions, metadata)
- **Data**: `data/*.json` (processed datasets, API responses)
- **Logs**: `logs/*.json` (training logs, diagnostics)

### 6. **Best Practices**
- Always use `indent=2` when writing JSON for readability in version control
- Validate JSON schemas for critical configs using a schema validator library
- Use `json.JSONDecodeError` exception handling for robustness
- Store datetime as ISO format strings (`.isoformat()`)
- Keep JSON files human-readable for code review

---

## Architecture
- `src/ai_engineering_lab/` — Main module (utilities, experiment runners)
- `Untitled.ipynb` — Exploratory notebooks
- `pyproject.toml` — Dependencies and project metadata
- `.python-version` — Python version pinning

## Development Notes
- Python environment: Uses VS Code Python extension with system environment manager
- Notebooks: Remember to restart kernel when dependencies change
- Tests: Keep test scripts in `src/ai_engineering_lab/` for easy discovery

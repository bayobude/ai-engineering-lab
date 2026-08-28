FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    scikit-learn \
    joblib \
    pydantic

EXPOSE 8000

CMD ["uvicorn", "ai_engineering_lab.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
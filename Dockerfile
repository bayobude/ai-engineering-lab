FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies
RUN uv sync --locked --no-dev --no-install-project

# Copy application source code
COPY src ./src

# Copy trained machine learning model
COPY models ./models

# Install the project
RUN uv sync --locked --no-dev

# Use the virtual environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "ai_engineering_lab.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
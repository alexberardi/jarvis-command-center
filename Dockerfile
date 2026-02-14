FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app

COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY alembic.ini ./
COPY alembic/ ./alembic

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

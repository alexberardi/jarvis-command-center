FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY alembic.ini ./
COPY alembic/ ./alembic

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

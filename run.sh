#!/bin/bash
export $(grep -v '^#' .env | xargs)

# Start virtual env
source venv/bin/activate

# run app
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-9998} --reload

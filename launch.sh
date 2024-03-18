#!/bin/sh

# Default to 1 worker if no argument is provided
NUM_WORKERS=${1:-1}

# Default to 20 chunks if no argument is provided
STREAM_CHUNK_SIZE=${2:-20}

python modeldownload.py
uvicorn tts_server:app --host 0.0.0.0 --port 6006 --workers $NUM_WORKERS --proxy-headers --env-file .env &
sleep 5
python script.py
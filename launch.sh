#!/bin/sh

# Default to 1 worker if no argument is provided
NUM_WORKERS=${1:-1}

python modeldownload.py
uvicorn tts_server:app --host 0.0.0.0 --port 6006 --workers $NUM_WORKERS --proxy-headers &
sleep 5
python script.py
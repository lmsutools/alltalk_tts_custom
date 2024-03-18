#!/bin/sh

# Default to 1 worker if no argument is provided
NUM_WORKERS=1
STREAM_CHUNK_SIZE=20

while getopts ":r:c:" opt; do
  case $opt in
    r) NUM_WORKERS="$OPTARG"
    ;;
    c) STREAM_CHUNK_SIZE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

python modeldownload.py
STREAM_CHUNK_SIZE=$STREAM_CHUNK_SIZE uvicorn tts_server:app --host 0.0.0.0 --port 6006 --workers $NUM_WORKERS --proxy-headers &
sleep 5
python script.py
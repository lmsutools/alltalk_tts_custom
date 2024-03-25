#!/bin/sh

# Default to 1 worker if no argument is provided for the number of workers
NUM_WORKERS=${1:-1}
# Default to mp3 if no argument is provided for the format
FORMAT=${2:-mp3}

# Download the model first
python modeldownload.py

# Check the format and start the appropriate server
if [ "$FORMAT" = "wav" ]; then
    echo "Launching WAV server"
    uvicorn tts_server_wav:app --host 0.0.0.0 --port 6006 --workers $NUM_WORKERS --proxy-headers &
else
    echo "Launching MP3 server"
    uvicorn tts_server_mp3:app --host 0.0.0.0 --port 6006 --workers $NUM_WORKERS --proxy-headers &
fi

# Wait for the server to start
sleep 5

# Run additional script
python script.py

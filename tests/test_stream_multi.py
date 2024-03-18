#<test_stream_multi.py> 
import argparse
import shutil
import sys
import time
from typing import Iterator
import requests
import threading
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the server URL from the environment variable
server_url = os.getenv("SERVER_URL")

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True

def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)

def tts(text, voice, language, server_url, output_file, request_number, first_chunk_times, lock) -> None:
    start = time.perf_counter()
    # Encode the text for URL
    encoded_text = requests.utils.quote(text)
    # Create the streaming URL
    streaming_url = f"{server_url}/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
    res = requests.get(streaming_url, stream=True)
    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)

    first_chunk_received = False
    for chunk in res.iter_content(chunk_size=512):
        if not first_chunk_received:
            end = time.perf_counter()
            first_chunk_time = end - start
            with lock:
                first_chunk_times.append(first_chunk_time)
            print(f"REQUEST # {request_number}")
            print(f"-> First chunk received after {first_chunk_time:.3f} seconds of the request being sent.", file=sys.stderr)
            print(f"-> Total character count: {len(text)}", file=sys.stderr)
            first_chunk_received = True

def send_requests(args):
    threads = []
    first_chunk_times = []
    lock = threading.Lock()
    start_time = time.perf_counter()  # Record the start time
    for i in range(args.requests):
        t = threading.Thread(target=tts, args=(args.text, args.voice, args.language, args.server_url, args.output_file, i + 1, first_chunk_times, lock))
        threads.append(t)
        t.start()
        time.sleep(1)  # Ramp-up of 1 second between each request

    for t in threads:
        t.join()

    total_first_chunk_time = sum(first_chunk_times)
    average_first_chunk_time = total_first_chunk_time / args.requests  # Calculate the average first chunk time

    print(f"TOTAL FIRST CHUNKS TIME: {total_first_chunk_time:.3f} seconds")
    print(f"AVERAGE FIRST CHUNKS TIMES: {average_first_chunk_time:.3f} seconds")

    end_time = time.perf_counter()  # Record the end time
    total_processing_time = end_time - start_time  # Calculate the total processing time
    print(f"TOTAL PROCESSING TIME: {total_processing_time:.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=("So If you encounter permission errors while installing packages, you can try running PowerShells." 
                 "On Wikipedia and other sites running on MediaWiki "
                 "So If you encounter permission errors while installing packages, you can try running PowerShells."
                 "So If you encounter errors while installing packages, you can try running PowerShells."),
        help="text input for TTS"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language to use default is 'en' (English)"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Save TTS output to given filename"
    )
    parser.add_argument(
        "--voice",
        default="female_01.wav",
        help="Voice to use for TTS"
    )
    # Use the server_url variable in your code
    parser.add_argument(
        "--server_url",
        default=server_url,
    )
    parser.add_argument(
        "-r", "--requests",
        type=int,
        default=1,
        help="Number of requests to send (default: 1)"
    )
    args = parser.parse_args()
    send_requests(args)
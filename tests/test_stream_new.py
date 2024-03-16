import argparse
import shutil
import subprocess
import sys
import time
from typing import Iterator
import requests
import tempfile
import os
import json

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True

def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)

def get_audio_info(filename: str) -> dict:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filename]
        output = subprocess.check_output(cmd).decode("utf-8")
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        print(f"Error getting audio info: {e}")
        return {}

def stream_ffplay(audio_stream, output_file=None, save=True):
    # Set the default output file name if not specified
    if output_file is None:
        output_file = "output.wav"
    
    # Construct the full path for the output file in the root folder
    # Assuming this script is located in the root folder of your project
    output_file_path = os.path.join(os.getcwd(), output_file)

    if not save:
        ffplay_cmd = ["ffplay", "-nodisp", "-autoexit", "-"]
    else:
        print("Saving to", output_file_path)
        # Use ffmpeg to save the audio; ensure ffmpeg is available in your environment
        ffplay_cmd = ["ffmpeg", "-i", "-", "-c:a", "pcm_s16le", "-f", "wav", "-y", output_file_path]

    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    first_chunk_received = False
    for chunk in audio_stream:
        if chunk is not None:
            if not first_chunk_received:
                first_chunk_time = time.perf_counter()
                print(f"First chunk played at {first_chunk_time} seconds")
                first_chunk_received = True
            ffplay_proc.stdin.write(chunk)

    # Close the ffplay stdin and wait for the process to finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()

    if save:
        print(f"Audio saved to {output_file_path}")


def tts(text, voice, language, server_url, output_file) -> Iterator[bytes]:
    start = time.perf_counter()
    # Encode the text for URL
    encoded_text = requests.utils.quote(text)
    # Create the streaming URL
    streaming_url = f"{server_url}/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}"
    res = requests.get(streaming_url, stream=True)
    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)
    first_chunk_received = False
    for chunk in res.iter_content(chunk_size=8192):
        if not first_chunk_received:
            end = time.perf_counter()
            print(f"-> First chunk received after {end-start:.3f} seconds of the request being sent.", file=sys.stderr)
            print(f"-> Total character count: {len(text)}", file=sys.stderr)
            first_chunk_received = True
        if chunk:
            yield chunk
    print(f"-> Total response time: {res.elapsed.total_seconds():.3f} seconds", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=("So If you encounter permission errors while installing packages, you can try running PowerShells. "
                 "So If you encounter permission errors while installing packages, you can try running PowerShells. "
                 "On Wikipedia and other sites running on MediaWiki "),
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
    parser.add_argument(
        "--server_url",
        default="https://f4aacd6da9151.notebooksb.jarvislabs.net",
    )
    args = parser.parse_args()

    audio = stream_ffplay(
        tts(
            args.text,
            args.voice,
            args.language,
            args.server_url,
            args.output_file
        ),
        args.output_file,
        save=bool(args.output_file)
    )
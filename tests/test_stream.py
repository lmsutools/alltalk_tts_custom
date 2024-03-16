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
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filename]
    output = subprocess.check_output(cmd).decode("utf-8")
    return json.loads(output)

def stream_ffplay(audio_stream, output_file, save=True):
    if not save:
        ffplay_cmd = ["ffplay", "-nodisp", "-probesize", "1024", "-autoexit", "-"]
    else:
        print("Saving to", output_file)
        ffplay_cmd = ["ffmpeg", "-probesize", "1024", "-i", "-", output_file]

    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    first_chunk_received = False
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for chunk in audio_stream:
            if chunk is not None:
                ffplay_proc.stdin.write(chunk)
                if not first_chunk_received:
                    temp_file.write(chunk)
                    temp_file.flush()
                    audio_info = get_audio_info(temp_file.name)
                    print("Audio Information (First Chunk):")
                    print(f"  Sample Rate: {audio_info['streams'][0]['sample_rate']} Hz")
                    print(f"  Codec: {audio_info['streams'][0]['codec_name']}")
                    first_chunk_received = True

    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()

    # Remove the temporary file
    os.unlink(temp_file.name)

def tts(text, voice, language, server_url, output_file) -> Iterator[bytes]:
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
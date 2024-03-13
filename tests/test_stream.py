
import argparse
import shutil
import subprocess
import sys
import time
from typing import Iterator
import requests

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True

def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)

def stream_ffplay(audio_stream, output_file, save=True):
    if not save:
        ffplay_cmd = ["ffplay", "-nodisp", "-probesize", "1024", "-autoexit", "-"]
    else:
        print("Saving to ", output_file)
        ffplay_cmd = ["ffmpeg", "-probesize", "1024", "-i", "-", output_file]
    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    for chunk in audio_stream:
        if chunk is not None:
            ffplay_proc.stdin.write(chunk)
    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()

def tts(text, voice, language, server_url, output_file) -> Iterator[bytes]:
    start = time.perf_counter()
    
    # Encode the text for URL
    encoded_text = requests.utils.quote(text)
    
    # Create the streaming URL
    streaming_url = f"{server_url}/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
    
    res = requests.get(streaming_url, stream=True)
    
    end = time.perf_counter()
    print(f"Time to make GET: {end-start}s", file=sys.stderr)
    
    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)
    
    first = True
    for chunk in res.iter_content(chunk_size=512):
        if first:
            end = time.perf_counter()
            print(f"Time to first chunk: {end-start}s", file=sys.stderr)
            first = False
        if chunk:
            yield chunk
    
    print("⏱️ response.elapsed:", res.elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", default="So If you encounter permission errors while installing packages, you can try running PowerShell. On Wikipedia and other sites running on MediaWiki, Special:Random can be used to access a random article in the main namespace; this feature is useful as a tool to generate a random article. Depending on your browser, it's also possible to load a random page. So If you encounter permission errors while installing packages, you can try running PowerShell. On Wikipedia and other sites running on MediaWiki, Special:Random can be used to access a random article in the main namespace;",
        help="text input for TTS"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language to use default is 'en' (English)"
    )
    parser.add_argument(
        "--output_file", default=None,
        help="Save TTS output to given filename"
    )
    parser.add_argument(
        "--voice", default="female_01.wav",
        help="Voice to use for TTS"
    )
    parser.add_argument(
        "--server_url", default="https://7ba301f55a0c1.notebooksc.jarvislabs.net",
       
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

#python .\tests\loadtest.py --num_requests 1 --text "So If you encounter permission errors while installing packages" --voice "female_01.wav"

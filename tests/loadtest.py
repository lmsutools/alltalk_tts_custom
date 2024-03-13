import argparse
import asyncio
import aiohttp
import time
import sys
from urllib.parse import quote

async def tts_request(session, text, voice, language, output_file):
    start_time = time.perf_counter()
    
    # Encode the text for URL
    encoded_text = quote(text)
    
    # Create the streaming URL
    streaming_url = f"https://bf13c60e2ff11.notebooksb.jarvislabs.net/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
    
    async with session.get(streaming_url) as res:
        end_time = time.perf_counter()
        response_time = end_time - start_time
        print(f"Response time: {response_time:.4f} seconds")
        
        if res.status != 200:
            print(f"Error: {res.status}")
            return
        
        audio_start_time = None
        async for chunk in res.content.iter_chunked(512):
            if audio_start_time is None:
                audio_start_time = time.perf_counter()
                print(f"Audio started playing at: {audio_start_time - start_time:.4f} seconds")
        
        audio_end_time = time.perf_counter()
        audio_duration = audio_end_time - audio_start_time
        print(f"Audio duration: {audio_duration:.4f} seconds")

async def load_test(num_requests, text, voice, language, output_file):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            task = asyncio.ensure_future(tts_request(session, text, voice, language, output_file))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_requests", type=int, default=10,
        help="Number of concurrent requests to send"
    )
    parser.add_argument(
        "--text", default="This is a test text for load testing.",
        help="Text input for TTS"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language to use, default is 'en' (English)"
    )
    parser.add_argument(
        "--output_file", default="output.wav",
        help="Output file name"
    )
    parser.add_argument(
        "--voice", default="female_01.wav",
        help="Voice to use for TTS"
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_test(args.num_requests, args.text, args.voice, args.language, args.output_file))
    end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"\nTotal time: {total_time:.4f} seconds")
    print(f"Requests per second: {args.num_requests / total_time:.2f}")
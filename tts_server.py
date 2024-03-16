# <tts_server.py>
import json
import time
import os
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import wave
import ffmpeg

##########################
#### Webserver Imports####
##########################
from fastapi import (
    FastAPI,
    Form,
    Request,
    Response,
    Depends,
    HTTPException,
)
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# ... (rest of the code remains the same)

# TTS VOICE GENERATION METHODS (called from voice_preview and output_modifer)
async def generate_audio(text, voice, language, temperature, repetition_penalty, output_file, streaming=False):
    # Get the async generator from the internal function
    response = generate_audio_internal(text, voice, language, temperature, repetition_penalty, output_file, streaming)
    # If streaming, then return the generator as-is, otherwise just exhaust it and return
    if streaming:
        return response
    async for _ in response:
        pass
    
async def generate_audio_internal(text, voice, language, temperature, repetition_penalty, output_file, streaming):
    global model
    if params["low_vram"] and device == "cpu":
        await switch_device()
    generate_start_time = time.time()  # Record the start time of generating TTS
    
    # XTTSv2 LOCAL & Xttsv2 FT Method
    if params["tts_method_xtts_local"] or tts_method_xtts_ft:
        print(f"[{params['branding']}TTSGen] {text}")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[f"{this_dir}/voices/{voice}"],
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )

        # Common arguments for both functions
        common_args = {
            "text": text,
            "language": language,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "temperature": float(temperature),
            "length_penalty": float(model.config.length_penalty),
            "repetition_penalty": float(repetition_penalty),
            "top_k": int(model.config.top_k),
            "top_p": float(model.config.top_p),
            "enable_text_splitting": True
        }

        # Determine the correct inference function and add streaming specific argument if needed
        inference_func = model.inference_stream if streaming else model.inference
        if streaming:
            common_args["stream_chunk_size"] = 10

        # Call the appropriate function
        output = inference_func(**common_args) 

        # Process the output based on streaming or non-streaming
        if streaming:
            # Streaming-specific operations
            file_chunks = []
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as vfout:
                vfout.setnchannels(1)
                vfout.setsampwidth(2)
                vfout.setframerate(24000)
                vfout.writeframes(b"")
            wav_buf.seek(0)

            # Encode WAV to MP3 on the fly using ffmpeg
            process = (
                ffmpeg
                .input('pipe:', format='wav')
                .output('pipe:', format='mp3', acodec='libmp3lame', audio_bitrate=128000)
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            process.stdin.write(wav_buf.read())
            yield process.stdout.read(1024)

            for i, chunk in enumerate(output):
                file_chunks.append(chunk)
                if isinstance(chunk, list):
                    chunk = torch.cat(chunk, dim=0)
                chunk = chunk.clone().detach().cpu().numpy()
                chunk = chunk[None, : int(chunk.shape[0])]
                chunk = np.clip(chunk, -1, 1)
                chunk = (chunk * 32767).astype(np.int16)
                process.stdin.write(chunk.tobytes())
                yield process.stdout.read(1024)

            process.stdin.close()
            process.wait()
        else:
            # Non-streaming-specific operation
            torchaudio.save(output_file, torch.tensor(output["wav"]).unsqueeze(0), 24000)

    # API LOCAL Methods
    elif params["tts_method_api_local"]:
        # Streaming only allowed for XTTSv2 local
        if streaming:
            raise ValueError("Streaming is only supported in XTTSv2 local")

        # Set the correct output path (different from the if statement)
        print(f"[{params['branding']}TTSGen] Using API Local")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
            temperature=temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )

    # API TTS
    elif params["tts_method_api_tts"]:
        # Streaming only allowed for XTTSv2 local
        if streaming:
            raise ValueError("Streaming is only supported in XTTSv2 local")

        print(f"[{params['branding']}TTSGen] Using API TTS")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
        )

    # Print Generation time and settings
    generate_end_time = time.time()  # Record the end time to generate TTS
    generate_elapsed_time = generate_end_time - generate_start_time
    print(
        f"[{params['branding']}TTSGen] \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{params['low_vram']} \033[94mDeepSpeed: \033[33m{params['deepspeed_activate']}\033[0m"
    )
    # Move model back to cpu system ram if needed.
    if params["low_vram"] and device == "cuda":
        await switch_device()
    return

# TTS VOICE GENERATION METHODS - generate TTS API
@app.route("/api/generate", methods=["POST"])
async def generate(request: Request):
    try:
        # Get parameters from JSON body
        data = await request.json()
        text = data["text"]
        voice = data["voice"]
        language = data["language"]
        temperature = data["temperature"]
        repetition_penalty = data["repetition_penalty"]
        output_file = data["output_file"]
        streaming = False
        # Generation logic
        response = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file, streaming)
        if streaming:
            return StreamingResponse(response, media_type="audio/wav")
        return JSONResponse(
            content={"status": "generate-success", "data": {"audio_path": output_file}}
        )
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


###################################################
#### POPULATE FILES LIST FROM VOICES DIRECTORY ####
###################################################
# List files in the "voices" directory
def list_files(directory):
    files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")
    ]
    return files

#############################
#### JSON CONFIG UPDATER ####
#############################

# Create an instance of Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=this_dir / "templates")

# Create a dependency to get the current JSON data
def get_json_data():
    with open(this_dir / "confignew.json", "r") as json_file:
        data = json.load(json_file)
    return data


# Define an endpoint function
@app.get("/settings")
async def get_settings(request: Request):
    wav_files = list_files(this_dir / "voices")
    # Render the template with the current JSON data and list of WAV files
    return templates.TemplateResponse(
        "generate_form.html",
        {
            "request": request,
            "data": get_json_data(),
            "modeldownload_model_path": modeldownload_model_path,
            "wav_files": wav_files,
        },
    )

# Define an endpoint to serve static files
app.mount("/static", StaticFiles(directory=str(this_dir / "templates")), name="static")

@app.post("/update-settings")
async def update_settings(
    request: Request,
    activate: bool = Form(...),
    autoplay: bool = Form(...),
    deepspeed_activate: bool = Form(...),
    delete_output_wavs: str = Form(...),
    ip_address: str = Form(...),
    language: str = Form(...),
    local_temperature: str = Form(...),
    local_repetition_penalty: str = Form(...),
    low_vram: bool = Form(...),
    tts_model_loaded: bool = Form(...),
    tts_model_name: str = Form(...),
    narrator_enabled: bool = Form(...),
    narrator_voice: str = Form(...),
    output_folder_wav: str = Form(...),
    port_number: str = Form(...),
    remove_trailing_dots: bool = Form(...),
    show_text: bool = Form(...),
    tts_method: str = Form(...),
    voice: str = Form(...),
    data: dict = Depends(get_json_data),
):
    # Update the settings based on the form values
    data["activate"] = activate
    data["autoplay"] = autoplay
    data["deepspeed_activate"] = deepspeed_activate
    data["delete_output_wavs"] = delete_output_wavs
    data["ip_address"] = ip_address
    data["language"] = language
    data["local_temperature"] = local_temperature
    data["local_repetition_penalty"] = local_repetition_penalty
    data["low_vram"] = low_vram
    data["tts_model_loaded"] = tts_model_loaded
    data["tts_model_name"] = tts_model_name
    data["narrator_enabled"] = narrator_enabled
    data["narrator_voice"] = narrator_voice
    data["output_folder_wav"] = output_folder_wav
    data["port_number"] = port_number
    data["remove_trailing_dots"] = remove_trailing_dots
    data["show_text"] = show_text
    data["tts_method_api_local"] = tts_method == "api_local"
    data["tts_method_api_tts"] = tts_method == "api_tts"
    data["tts_method_xtts_local"] = tts_method == "xtts_local"
    data["voice"] = voice

    # Save the updated settings back to the JSON file
    with open(this_dir / "confignew.json", "w") as json_file:
        json.dump(data, json_file)

    # Redirect to the settings page to display the updated settings
    return RedirectResponse(url="/settings", status_code=303)


##################################
#### SETTINGS PAGE DEMO VOICE ####
##################################

@app.get("/tts-demo-request", response_class=StreamingResponse)
async def tts_demo_request_streaming(text: str, voice: str, language: str, output_file: str):
    try:
        output_file_path = this_dir / "outputs" / output_file
        stream = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/tts-demo-request", response_class=JSONResponse)
async def tts_demo_request(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)


#####################
#### Audio feeds ####
#####################

# Gives web access to the output files
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = this_dir / "outputs" / filename
    return FileResponse(audio_path)

@app.get("/audiocache/{filename}")
async def get_audio(filename: str):
    audio_path = Path("outputs") / filename
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    response = FileResponse(
        path=audio_path,
        media_type='audio/wav',
        filename=filename
    )
    # Set caching headers
    response.headers["Cache-Control"] = "public, max-age=604800"  # Cache for one week
    response.headers["ETag"] = str(audio_path.stat().st_mtime)  # Use the file's last modified time as a simple ETag

    return response

#########################
#### VOICES LIST API ####
#########################
# Define the new endpoint
@app.get("/api/voices")
async def get_voices():
    wav_files = list_files(this_dir / "voices")
    return {"voices": wav_files}

###########################
#### PREVIEW VOICE API ####
###########################
@app.post("/api/previewvoice/", response_class=JSONResponse)
async def preview_voice(request: Request, voice: str = Form(...)):
    try:
        # Hardcoded settings
        language = "en"
        output_file_name = "api_preview_voice"

        # Clean the voice filename for inclusion in the text
        clean_voice_filename = re.sub(r'\.wav$', '', voice.replace(' ', '_'))
        clean_voice_filename = re.sub(r'[^a-zA-Z0-9]', ' ', clean_voice_filename)
        
        # Generate the audio
        text = f"Hello, this is a preview of voice {clean_voice_filename}."

        # Generate the audio
        output_file_path = this_dir / "outputs" / f"{output_file_name}.wav"
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)

        # Generate the URL
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'

        # Return the response with both local file path and URL
        return JSONResponse(
            content={
                "status": "generate-success",
                "output_file_path": str(output_file_path),
                "output_file_url": str(output_file_url),
            },
            status_code=200,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

########################
#### GENERATION API ####
########################
import html
import re
import uuid
import numpy as np
import soundfile as sf
import sys
import hashlib

##############################
#### Streaming Generation ####
##############################

@app.get("/api/tts-generate-streaming", response_class=StreamingResponse)
async def tts_generate_streaming(text: str, voice: str, language: str, output_file: str):
    try:
        output_file_path = this_dir / "outputs" / output_file
        stream = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/api/tts-generate-streaming", response_class=JSONResponse)
async def tts_generate_streaming(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

##############################
#### Standard Generation ####
##############################

# Check for PortAudio library on Linux
try:
    import sounddevice as sd
    sounddevice_installed=True
except OSError:
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m PortAudio library not found. If you wish to play TTS in standalone mode through the API suite")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m please install PortAudio. This will not affect any other features or use of Alltalk.")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m If you don't know what the API suite is, then this message is nothing to worry about.")
    sounddevice_installed=False
    if sys.platform.startswith('linux'):
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m On Linux, you can use the following command to install PortAudio:")
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m sudo apt-get install portaudio19-dev")

from typing import Union, Dict
from pydantic import BaseModel, ValidationError, Field

def play_audio(file_path, volume):
    data, fs = sf.read(file_path)
    sd.play(volume * data, fs)
    sd.wait()

class Request(BaseModel):
    # Define the structure of the 'Request' class if needed
    pass

class JSONInput(BaseModel):
    text_input: str = Field(..., max_length=2000, description="text_input needs to be 2000 characters or less.")
    text_filtering: str = Field(..., pattern="^(none|standard|html)$", description="text_filtering needs to be 'none', 'standard' or 'html'.")
    character_voice_gen: str = Field(..., pattern="^.*\.wav$", description="character_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    narrator_enabled: bool = Field(..., description="narrator_enabled needs to be true or false.")
    narrator_voice_gen: str = Field(..., pattern="^.*\.wav$", description="narrator_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    text_not_inside: str = Field(..., pattern="^(character|narrator)$", description="text_not_inside needs to be 'character' or 'narrator'.")
    language: str = Field(..., pattern="^(ar|zh-cn|cs|nl|en|fr|de|hu|it|ja|ko|pl|pt|ru|es|tr)$", description="language needs to be one of the following ar|zh-cn|cs|nl|en|fr|de|hu|it|ja|ko|pl|pt|ru|es|tr.")
    output_file_name: str = Field(..., pattern="^[a-zA-Z0-9_]+$", description="output_file_name needs to be the name without any special characters or file extension e.g. 'filename'")
    output_file_timestamp: bool = Field(..., description="output_file_timestamp needs to be true or false.")
    autoplay: bool = Field(..., description="autoplay needs to be a true or false value.")
    autoplay_volume: float = Field(..., ge=0.1, le=1.0, description="autoplay_volume needs to be from 0.1 to 1.0")

    @classmethod
    def validate_autoplay_volume(cls, value):
        if not (0.1 <= value <= 1.0):
            raise ValueError("Autoplay volume must be between 0.1 and 1.0")
        return value


class TTSGenerator:
    @staticmethod
    def validate_json_input(json_data: Union[Dict, str]) -> Union[None, str]:
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            JSONInput(**json_data)
            return None  # JSON is valid
        except ValidationError as e:
            return str(e)

def process_text(text):
    # Normalize HTML encoded quotes
    text = html.unescape(text)
    # Replace ellipsis with a single dot
    text = re.sub(r'\.{3,}', '.', text)
    # Pattern to identify combined narrator and character speech
    combined_pattern = r'(\*[^*"]+\*|"[^"*]+")'
    # List to hold parts of speech along with their type
    ordered_parts = []
    # Track the start of the next segment
    start = 0
    # Find all matches
    for match in re.finditer(combined_pattern, text):
        # Add the text before the match, if any, as ambiguous
        if start < match.start():
            ambiguous_text = text[start:match.start()].strip()
            if ambiguous_text:
                ordered_parts.append(('ambiguous', ambiguous_text))
        # Add the matched part as either narrator or character
        matched_text = match.group(0)
        if matched_text.startswith('*') and matched_text.endswith('*'):
            ordered_parts.append(('narrator', matched_text.strip('*').strip()))
        elif matched_text.startswith('"') and matched_text.endswith('"'):
            ordered_parts.append(('character', matched_text.strip('"').strip()))
        else:
            # In case of mixed or improperly formatted parts
            if '*' in matched_text:
                ordered_parts.append(('narrator', matched_text.strip('*').strip('"')))
            else:
                ordered_parts.append(('character', matched_text.strip('"').strip('*')))
        # Update the start of the next segment
        start = match.end()
    # Add any remaining text after the last match as ambiguous
    if start < len(text):
        ambiguous_text = text[start:].strip()
        if ambiguous_text:
            ordered_parts.append(('ambiguous', ambiguous_text))
    return ordered_parts

def standard_filtering(text_input):
    text_output = (text_input
                        .replace("***", "")
                        .replace("**", "")
                        .replace("*", "")
                        .replace("\n\n", "\n")
                        .replace("&#x27;", "'")
                        )
    return text_output

def combine(output_file_timestamp, output_file_name, audio_files):
    audio = np.array([])
    sample_rate = None
    try:
        for audio_file in audio_files:
            audio_data, current_sample_rate = sf.read(audio_file)
            if audio.size == 0:
                audio = audio_data
                sample_rate = current_sample_rate
            elif sample_rate == current_sample_rate:
                audio = np.concatenate((audio, audio_data))
            else:
                raise ValueError("Sample rates of input files are not consistent.")
    except Exception as e:
        # Handle exceptions (e.g., file not found, invalid audio format)
        return None, None
    if output_file_timestamp:
        timestamp = int(time.time())
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_{timestamp}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}_combined.wav'
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_{timestamp}_combined.wav'
    else:
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_combined.wav'
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_combined.wav'
    try:
        sf.write(output_file_path, audio, samplerate=sample_rate)
        # Clean up unnecessary files
        for audio_file in audio_files:
            os.remove(audio_file)
    except Exception as e:
        # Handle exceptions (e.g., failed to write output file)
        return None, None
    return output_file_path, output_file_url, output_cache_url

# Generation API (separate from text-generation-webui)
@app.post("/api/tts-generate", response_class=JSONResponse)
async def tts_generate(
    text_input: str = Form(...),
    text_filtering: str = Form(...),
    character_voice_gen: str = Form(...),
    narrator_enabled: bool = Form(...),
    narrator_voice_gen: str = Form(...),
    text_not_inside: str = Form(...),
    language: str = Form(...),
    output_file_name: str = Form(...),
    output_file_timestamp: bool = Form(...),
    autoplay: bool = Form(...),
    autoplay_volume: float = Form(...),
    streaming: bool = Form(False),
):
    try:
        json_input_data = {
            "text_input": text_input,
            "text_filtering": text_filtering,
            "character_voice_gen": character_voice_gen,
            "narrator_enabled": narrator_enabled,
            "narrator_voice_gen": narrator_voice_gen,
            "text_not_inside": text_not_inside,
            "language": language,
            "output_file_name": output_file_name,
            "output_file_timestamp": output_file_timestamp,
            "autoplay": autoplay,
            "autoplay_volume": autoplay_volume,
            "streaming": streaming,
        }
        JSONresult = TTSGenerator.validate_json_input(json_input_data)
        if JSONresult is None:
            pass
        else:
            return JSONResponse(content={"error": JSONresult}, status_code=400)
        if narrator_enabled:
            processed_parts = process_text(text_input)
            audio_files_all_paragraphs = []
            for part_type, part in processed_parts:
                # Skip parts that are too short
                if len(part.strip()) <= 3:
                    continue
                # Determine the voice to use based on the part type
                if part_type == 'narrator':
                    voice_to_use = narrator_voice_gen
                    print(f"[{params['branding']}TTSGen] \033[92mNarrator\033[0m")  # Green
                elif part_type == 'character':
                    voice_to_use = character_voice_gen
                    print(f"[{params['branding']}TTSGen] \033[36mCharacter\033[0m")  # Yellow
                else:
                    # Handle ambiguous parts based on user preference
                    voice_to_use = character_voice_gen if text_not_inside == "character" else narrator_voice_gen
                    voice_description = "\033[36mCharacter (Text-not-inside)\033[0m" if text_not_inside == "character" else "\033[92mNarrator (Text-not-inside)\033[0m"
                    print(f"[{params['branding']}TTSGen] {voice_description}")
                # Replace multiple exclamation marks, question marks, or other punctuation with a single instance
                cleaned_part = re.sub(r'([!?.])\1+', r'\1', part)
                # Further clean to remove any other unwanted characters
                cleaned_part = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$', '', cleaned_part)
                # Remove all newline characters (single or multiple)
                cleaned_part = re.sub(r'\n+', ' ', cleaned_part)
                output_file = this_dir / "outputs" / f"{output_file_name}_{uuid.uuid4()}_{int(time.time())}.wav"
                output_file_str = output_file.as_posix()
                response = await generate_audio(cleaned_part, voice_to_use, language,temperature, repetition_penalty, output_file_str, streaming)
                audio_path = output_file_str
                audio_files_all_paragraphs.append(audio_path)
            # Combine audio files across paragraphs
            output_file_path, output_file_url, output_cache_url = combine(output_file_timestamp, output_file_name, audio_files_all_paragraphs)
        else:
            if output_file_timestamp:
                timestamp = int(time.time())
                # Generate a standard UUID
                original_uuid = uuid.uuid4()
                # Hash the UUID using SHA-256
                hash_object = hashlib.sha256(str(original_uuid).encode())
                hashed_uuid = hash_object.hexdigest()
                # Truncate to the desired length, for example, 16 characters
                short_uuid = hashed_uuid[:5]
                output_file_path = this_dir / "outputs" / f"{output_file_name}_{timestamp}{short_uuid}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}{short_uuid}.wav'
                output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_{timestamp}{short_uuid}.wav'
            else:
                output_file_path = this_dir / "outputs" / f"{output_file_name}.wav"
                output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}.wav'
                output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}.wav'
            if text_filtering == "html":
                cleaned_string = html.unescape(standard_filtering(text_input))
                cleaned_string = re.sub(r'([!?.])\1+', r'\1', text_input)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$', '', cleaned_string)
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            elif text_filtering == "standard":
                cleaned_string = re.sub(r'([!?.])\1+', r'\1', text_input)
                # Further clean to remove any other unwanted characters
                cleaned_string = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$', '', cleaned_string)
                # Remove all newline characters (single or multiple)
                cleaned_string = re.sub(r'\n+', ' ', cleaned_string)
            else:
                cleaned_string = text_input
            response = await generate_audio(cleaned_string, character_voice_gen, language, temperature, repetition_penalty, output_file_path, streaming)
        if sounddevice_installed == False or streaming == True:
            autoplay = False
        if autoplay:
            play_audio(output_file_path, autoplay_volume)       
        if streaming:
            return StreamingResponse(response, media_type="audio/wav")
        return JSONResponse(content={"status": "generate-success", "output_file_path": str(output_file_path), "output_file_url": str(output_file_url), "output_cache_url": str(output_cache_url)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "generate-failure", "error": "An error occurred"}, status_code=500)


##########################
#### Current Settings ####
##########################
# Define the available models
models_available = [
    {"name": "Coqui", "model_name": "API TTS"},
    {"name": "Coqui", "model_name": "API Local"},
    {"name": "Coqui", "model_name": "XTTSv2 Local"}
]

@app.get('/api/currentsettings')
def get_current_settings():
    # Determine the current model loaded
    if params["tts_method_api_tts"]:
        current_model_loaded = "API TTS"
    elif params["tts_method_api_local"]:
        current_model_loaded = "API Local"
    elif params["tts_method_xtts_local"]:
        current_model_loaded = "XTTSv2 Local"
    else:
        current_model_loaded = None  # or a default value if no method is active

    settings = {
        "models_available": models_available,
        "current_model_loaded": current_model_loaded,
        "deepspeed_available": deepspeed_available,
        "deepspeed_status": params["deepspeed_activate"],
        "low_vram_status": params["low_vram"],
        "finetuned_model": finetuned_model
    }
    return settings  # Automatically converted to JSON by Fas

#############################
#### Word Add-in Sharing ####
#############################
# Mount the static files from the 'word_addin' directory
app.mount("/api/word_addin", StaticFiles(directory=os.path.join(this_dir / 'templates' / 'word_addin')), name="word_addin")

###################################################
#### Webserver Startup & Initial model Loading ####
###################################################

# Get the admin interface template
template = templates.get_template("admin.html")
# Render the template with the dynamic values
rendered_html = template.render(params=params)

###############################
#### Internal script ready ####
###############################
@app.get("/ready")
async def ready():
    return Response("Ready endpoint")

############################
#### External API ready ####
############################
@app.get("/api/ready")
async def ready():
    return Response("Ready")

@app.get("/")
async def read_root():
    return HTMLResponse(content=rendered_html, status_code=200)

# Start Uvicorn Webserver
host_parameter = params["ip_address"]
port_parameter = int(params["port_number"])

if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    num_workers = LOGGING_CONFIG["formatters"]["default"]["use_colors"]
    print(f"[{params['branding']}Startup] \033[94mStarting Uvicorn with {num_workers} worker(s)\033[0m")

    uvicorn.run(app, host=host_parameter, port=port_parameter, log_level="warning")
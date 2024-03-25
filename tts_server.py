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
from pydub import AudioSegment

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

###########################
#### STARTUP VARIABLES ####
###########################
# STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()
# STARTUP VARIABLE - Set "device" to cuda if exists, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
# STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / "languages.json", encoding="utf8") as f:
    languages = json.load(f)
# Base setting for a possible FineTuned model existing and the loader being available
tts_method_xtts_ft = False

#################################################################
#### LOAD PARAMS FROM confignew.json - REQUIRED FOR BRANDING ####
#################################################################
# Load config file and get settings
def load_config(file_path):
    with open(file_path, "r") as configfile_path:
        configfile_data = json.load(configfile_path)
    return configfile_data


# Define the path to the confignew.json file
configfile_path = this_dir / "confignew.json"

# Load confignew.json and assign it to a different variable (config_data)
params = load_config(configfile_path)
# check someone hasnt enabled lowvram on a system thats not cuda enabled
params["low_vram"] = "false" if not torch.cuda.is_available() else params["low_vram"]

# Load values for temperature and repetition_penalty
temperature = params["local_temperature"]
repetition_penalty = params["local_repetition_penalty"]

# Define the path to the JSON file
config_file_path = this_dir / "modeldownload.json"

#############################################
#### LOAD PARAMS FROM MODELDOWNLOAD.JSON ####
############################################
# This is used only in the instance that someone has changed their model path
# Define the path to the JSON file
modeldownload_config_file_path = this_dir / "modeldownload.json"

# Check if the JSON file exists
if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, "r") as modeldownload_config_file:
        modeldownload_settings = json.load(modeldownload_config_file)

    # Extract settings from the loaded JSON
    modeldownload_base_path = Path(modeldownload_settings.get("base_path", ""))
    modeldownload_model_path = Path(modeldownload_settings.get("model_path", ""))
else:
    # Default settings if the JSON file doesn't exist or is empty
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m modeldownload.config is missing so please re-download it and save it in the alltalk_tts main folder."
    )

##################################################
#### Check to see if a finetuned model exists ####
##################################################
# Set the path to the directory
trained_model_directory = this_dir / "models" / "trainedmodel"
# Check if the directory "trainedmodel" exists
finetuned_model = trained_model_directory.exists()
# If the directory exists, check for the existence of the required files
if finetuned_model:
    required_files = ["model.pth", "config.json", "vocab.json"]
    finetuned_model = all((trained_model_directory / file).exists() for file in required_files)

########################
#### STARTUP CHECKS ####
########################
try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    print(
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the alltalk_tts extension.",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Linux / Mac:\npip install -r extensions/alltalk_tts/requirements.txt\n",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m Windows:\npip install -r extensions\\alltalk_tts\\requirements.txt\n",
        f"[{params['branding']}Startup] \033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the cmd_ script. On Windows, that's cmd_windows.bat."
    )
    raise

# DEEPSPEED Import - Check for DeepSpeed and import it if it exists
deepspeed_available = False
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    pass
if deepspeed_available:
    print(f"[{params['branding']}Startup] DeepSpeed \033[93mDetected\033[0m")
    print(f"[{params['branding']}Startup] Activate DeepSpeed in {params['branding']}settings")
else:
    print(f"[{params['branding']}Startup] DeepSpeed \033[93mNot Detected\033[0m. See https://github.com/microsoft/DeepSpeed")


@asynccontextmanager
async def startup_shutdown(no_actual_value_it_demanded_something_be_here):
    await setup()
    yield
    # Shutdown logic


# Create FastAPI app with lifespan
app = FastAPI(lifespan=startup_shutdown)
# Allow all origins, and set other CORS options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
# MODEL LOADERS Picker For API TTS, API Local, XTTSv2 Local, XTTSv2 FT
async def setup():
    global device
    # Set a timer to calculate load times
    generate_start_time = time.time()  # Record the start time of loading the model
    # Start loading the correct model as set by "tts_method_api_tts", "tts_method_api_local" or "tts_method_xtts_local" being True/False
    if params["tts_method_api_tts"]:
        print(
            f"[{params['branding']}Model] \033[94mAPI TTS Loading\033[0m {params['tts_model_name']} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await api_load_model()
    elif params["tts_method_api_local"]:
        print(
            f"[{params['branding']}Model] \033[94mAPI Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await api_manual_load_model()
    elif params["tts_method_xtts_local"]:
        print(
            f"[{params['branding']}Model] \033[94mXTTSv2 Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await xtts_manual_load_model()
    elif tts_method_xtts_ft:
        print(
            f"[{params['branding']}Model] \033[94mXTTSv2 FT Loading\033[0m /models/fintuned/model.pth \033[94minto\033[93m",
            device,
            "\033[0m",
        )
        model = await xtts_ft_manual_load_model()
    # Create an end timer for calculating load times
    generate_end_time = time.time()
    # Calculate start time minus end time
    generate_elapsed_time = generate_end_time - generate_start_time
    # Print out the result of the load time
    print(
        f"[{params['branding']}Model] \033[94mModel Loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m"
    )
    # Set "tts_model_loaded" to true
    params["tts_model_loaded"] = True
    # Set the output path for wav files
    output_directory = this_dir / params["output_folder_wav_standalone"]
    output_directory.mkdir(parents=True, exist_ok=True)
    #Path(f'this_folder/outputs/').mkdir(parents=True, exist_ok=True)


# MODEL LOADER For "API TTS"
async def api_load_model():
    global model
    model = TTS(params["tts_model_name"]).to(device)
    return model


# MODEL LOADER For "API Local"
async def api_manual_load_model():
    global model
    # check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        model = TTS(
            model_path=this_dir / "models" / modeldownload_model_path,
            config_path=this_dir / "models" / modeldownload_model_path / "config.json",
        ).to(device)
    else:
        print(
            f"[{params['branding']}Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:",
            modeldownload_base_path / modeldownload_model_path,
        )
        model = TTS(
            model_path=modeldownload_base_path / modeldownload_model_path,
            config_path=modeldownload_base_path / modeldownload_model_path / "config.json",
        ).to(device)
    return model


# MODEL LOADER For "XTTSv2 Local"
async def xtts_manual_load_model():
    global model
    config = XttsConfig()
    # check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        config_path = this_dir / "models" / modeldownload_model_path / "config.json"
        vocab_path_dir = this_dir / "models" / modeldownload_model_path / "vocab.json"
        checkpoint_dir = this_dir / "models" / modeldownload_model_path
    else:
        print(
            f"[{params['branding']}Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:",
            modeldownload_base_path / modeldownload_model_path,
        )
        config_path = modeldownload_base_path / modeldownload_model_path / "config.json"
        vocab_path_dir = modeldownload_base_path / modeldownload_model_path / "vocab.json"
        checkpoint_dir = modeldownload_base_path / modeldownload_model_path
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(vocab_path_dir),
        use_deepspeed=params["deepspeed_activate"],
    )
    model.to(device)
    return model

# MODEL LOADER For "XTTSv2 FT"
async def xtts_ft_manual_load_model():
    global model
    config = XttsConfig()
    config_path = this_dir / "models" / "trainedmodel" / "config.json"
    vocab_path_dir = this_dir / "models" / "trainedmodel" / "vocab.json"
    checkpoint_dir = this_dir / "models" / "trainedmodel"
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(vocab_path_dir),
        use_deepspeed=params["deepspeed_activate"],
    )
    model.to(device)
    return model

# MODEL UNLOADER
async def unload_model(model):
    print(f"[{params['branding']}Model] \033[94mUnloading model \033[0m")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    params["tts_model_loaded"] = False
    return None


# MODEL - Swap model based on Gradio selection API TTS, API Local, XTTSv2 Local
async def handle_tts_method_change(tts_method):
    global model
    global tts_method_xtts_ft
    # Update the params dictionary based on the selected radio button
    print(
        f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
    )
    # Set other parameters to False
    if tts_method == "API TTS":
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_tts"] = True
        params["deepspeed_activate"] = False
        tts_method_xtts_ft = False
    elif tts_method == "API Local":
        params["tts_method_api_tts"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_local"] = True
        params["deepspeed_activate"] = False
        tts_method_xtts_ft = False
    elif tts_method == "XTTSv2 Local":
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        tts_method_xtts_ft = False
    elif tts_method == "XTTSv2 FT":
        tts_method_xtts_ft = True
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False

    # Unload the current model
    model = await unload_model(model)

    # Load the correct model based on the updated params
    await setup()


# MODEL WEBSERVER- API Swap Between Models
@app.route("/api/reload", methods=["POST"])
async def reload(request: Request):
    tts_method = request.query_params.get("tts_method")
    if tts_method not in ["API TTS", "API Local", "XTTSv2 Local", "XTTSv2 FT"]:
        return {"status": "error", "message": "Invalid TTS method specified"}
    await handle_tts_method_change(tts_method)
    return Response(
        content=json.dumps({"status": "model-success"}), media_type="application/json"
    )



########################
#### TTS GENERATION ####
########################

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
            common_args["stream_chunk_size"] = 20

        # Call the appropriate function with error handling
        try:
            output = inference_func(**common_args)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"[{params['branding']}TTSGen] \033[91mError:\033[0m {e}")
            print(f"[{params['branding']}TTSGen] \033[91mClearing GPU memory and retrying...\033[0m")
            # Retry the inference with reduced batch size
            common_args["stream_chunk_size"] = 10
            output = inference_func(**common_args)

        # Process the output based on streaming or non-streaming
        if streaming:
            # Streaming-specific operations
            for i, chunk in enumerate(output):
                if isinstance(chunk, list):
                    chunk = torch.cat(chunk, dim=0)
                chunk = chunk.squeeze().cpu().numpy()
                chunk = np.clip(chunk, -1, 1)
                chunk = (chunk * 32767).astype(np.int16)

                # Convert the chunk to MP3 format using ffmpeg-python
                process = (
                    ffmpeg
                    .input('pipe:', format='s16le', ar=24000, ac=1)
                    .output('pipe:', format='mp3', audio_bitrate='128k', acodec='libmp3lame')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stdout=True)
                )
                process.stdin.write(chunk.tobytes())
                process.stdin.close()
                mp3_data = process.stdout.read()

                yield mp3_data
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



##########################
#### Current Settings ####
##########################
# Define the available models
models_available = [
    {"name": "Coqui", "model_name": "API TTS"},
    {"name": "Coqui", "model_name": "API Local"},
    {"name": "Coqui", "model_name": "XTTSv2 Local"}
]


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
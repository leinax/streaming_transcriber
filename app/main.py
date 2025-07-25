import os
import uuid
import tempfile
import concurrent.futures

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import yt_dlp

# App config
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Thread pool executor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Memoria temporal para resultados
results = {}

# ----- Funciones principales -----
def download_audio(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_file = os.path.join(temp_dir, "audio.wav")
    return audio_file

def transcribe_url(url: str) -> str:
    try:
        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "default"
        model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)

        audio_path = download_audio(url)
        audio = whisperx.load_audio(audio_path)
        audio = whisperx.pad_or_trim(audio)
        result = model.transcribe(audio, language=None)
        return result.get("text", "").strip() or "[Sin texto detectado]"
    except Exception as e:
        return f"Error al transcribir: {str(e)}"

def background_transcription(url: str, task_id: str):
    text = transcribe_url(url)
    results[task_id] = text

# ----- Endpoints -----
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_form(request: Request, url: str = Form(...)):
    transcription = transcribe_url(url)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "transcription": transcription
    })

class TranscribeRequest(BaseModel):
    url: str

@app.post("/api/transcribe")
async def transcribe_api(payload: TranscribeRequest):
    task_id = str(uuid.uuid4())
    results[task_id] = "processing"
    executor.submit(background_transcription, payload.url, task_id)
    return {"task_id": task_id}

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in results:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    result = results[task_id]
    if result == "processing":
        return {"status": "processing"}
    
    return {"status": "completed", "transcription": result}
import os
import tempfile
import subprocess

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import whisperx
import yt_dlp

# Configurar directorios
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de transcripción
device = "cuda" if whisperx.utils.get_device() == "cuda" else "cpu"
model = whisperx.load_model("large-v3", device=device, compute_type="float16" if device == "cuda" else "default")

def download_audio(url: str) -> str:
    """Descarga el audio desde la URL y devuelve el path al archivo WAV."""
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
        audio_path = download_audio(url)
        audio = whisperx.load_audio(audio_path)
        audio = whisperx.pad_or_trim(audio)
        result = model.transcribe(audio, language=None)
        text = result.get("text", "")
        return text.strip() or "[Sin texto detectado]"
    except Exception as e:
        return f"Error al transcribir: {str(e)}"

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
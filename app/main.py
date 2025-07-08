import os
import tempfile

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import yt_dlp

# Configurar app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "default"
        model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)

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

from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import uuid
import os

app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/transcribe")
async def transcribe_audio(request: URLRequest):
    video_url = request.url
    temp_id = str(uuid.uuid4())
    audio_path = f"/tmp/{temp_id}.mp3"

    try:
        # Download audio with yt-dlp
        subprocess.run([
            "yt-dlp", "-f", "bestaudio", "--extract-audio",
            "--audio-format", "mp3", "-o", audio_path, video_url
        ], check=True)

        # Run WhisperX transcription (requires GPU ideally)
        import whisperx
        model = whisperx.load_model("large-v2", device="cpu", compute_type="int8")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, language=None)

        text = ""
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Speaker")
            text += f"{speaker}: {segment['text']}\n"

        return {"transcription": text}

    except subprocess.CalledProcessError as e:
        return {"error": "Error downloading or processing audio."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

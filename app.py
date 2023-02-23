from whisper.audio import SAMPLE_RATE
from fastapi import FastAPI, File
import numpy as np
import subprocess
import whisper
import ffmpeg


app = FastAPI()
model_name = "medium.en"  # Yes, this is hardcoded.
model = whisper.load_model(model_name)


@app.get("/")
def root():
    return "shoo"


@app.post("/transcribe")
def transcribe(wav: bytes = File()):
    return {"result": model.transcribe(resample(wav))["text"]}


def resample(file):
    args = (
        ffmpeg.input("-", threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
        .get_args()
    )
    out, _ = subprocess.Popen(["ffmpeg"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(file)
    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    out = out / out.max()
    return out

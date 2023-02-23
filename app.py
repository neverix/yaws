from fastapi import FastAPI, UploadFile
import soundfile as sf
import whisper


app = FastAPI()
model_name = "medium.en"  # Yes, this is hardcoded.
model = whisper.load_model(model_name)


@app.get("/")
def root():
    return "shoo"


@app.post("/transcribe")
def transcribe(wav: UploadFile):
    file = wav.file
    y, sr = sf.read(file, dtype="float32")
    assert sr == 48_000  # For StereoKit
    abc = y[::3], y[1::3], y[2::3]
    ml = min(map(len, abc))
    y = sum(a[:ml] for a in abc) / 3  # DIY resampling! (Have I mentioned that this is not production-ready)
    return {"result": model.transcribe(y)["text"]}

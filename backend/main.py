"""FastAPI backend: /predict, /history, /health - and it also serves the frontend."""
import io
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .gradcam import GradCAM, overlay_heatmap
from .model_utils import CLASSES, MODEL_PATH, get_eval_transform, load_model

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "history.db"
MAX_BYTES = 10 * 1024 * 1024        # 10 MB upload limit
LOW_CONFIDENCE = 0.75               # below this we warn the user

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_eval_transform()
lock = threading.Lock()             # Grad-CAM uses shared state, so one request at a time
state = {"model": None, "cam": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT,
            confidence REAL, created_at TEXT)""")
    if MODEL_PATH.exists():
        state["model"] = load_model(MODEL_PATH, device)
        state["cam"] = GradCAM(state["model"], state["model"].layer4)
        print(f"Model loaded on {device}.")
    else:
        print("WARNING: no trained model found. Run `python train.py` first.")
    yield


app = FastAPI(title="Pneumonia Detector API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "device": str(device)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "Model not found. Train it first with: python train.py")

    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Please upload a valid image file (JPG or PNG).")

    x = transform(image).unsqueeze(0).to(device)
    with lock:
        cam, logits = state["cam"](x)

    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    idx = max(range(len(CLASSES)), key=lambda i: probs[i])
    label, confidence = CLASSES[idx], probs[idx]

    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO history (filename, label, confidence, created_at) VALUES (?,?,?,?)",
                    (file.filename, label, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {c: round(p, 4) for c, p in zip(CLASSES, probs)},
        "low_confidence": confidence < LOW_CONFIDENCE,
        "heatmap": overlay_heatmap(cam, image),
    }


@app.get("/history")
def history(limit: int = 20):
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT filename, label, confidence, created_at FROM history "
                           "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"filename": r[0], "label": r[1], "confidence": r[2], "created_at": r[3]} for r in rows]


# Serve the website at http://127.0.0.1:8000  (must come AFTER the API routes)
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")

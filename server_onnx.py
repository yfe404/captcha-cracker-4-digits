#!/usr/bin/env python3
import io, os, base64
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
import onnxruntime as ort

# ---------------------- CONFIG ----------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/resnet34_4head.onnx"))
DEVICE     = os.getenv("DEVICE", "cpu").lower()   # "cpu" or "cuda"
IMG_SIZE   = int(os.getenv("IMG_SIZE", "320"))
TTA_ANGLES = tuple(int(a) for a in os.getenv("TTA_ANGLES", "-4,0,4").split(",") if a)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
# ---------------------------------------------------

def letterbox(img: Image.Image, size: int = 320, bg: int = 255) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    out = Image.new("RGB", (size, size), (bg, bg, bg))
    out.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return out

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """grayscale->RGB, letterbox, normalize to ImageNet, return CHW float32"""
    img = img.convert("L").convert("RGB")
    img = letterbox(img, IMG_SIZE, bg=255)
    arr = np.asarray(img).astype(np.float32) / 255.0       # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                           # CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ORT session
providers = ["CPUExecutionProvider"]
if DEVICE == "cuda":
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]  # expect 4 outputs: head0..head3

app = FastAPI(title="4-digit OCR (ONNX Runtime)")

class B64Item(BaseModel):
    image_b64: str
    tta: Optional[bool] = False

class B64Batch(BaseModel):
    items: List[B64Item]

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": DEVICE, "providers": providers}

def run_onnx(batch: np.ndarray) -> List[np.ndarray]:
    """batch: N x 3 x H x W -> list of 4 arrays [N,10]"""
    outputs = sess.run(out_names, {in_name: batch})
    # Ensure float32
    return [o.astype(np.float32) for o in outputs]

def predict_pil(img: Image.Image, tta: bool = False, angles: Tuple[int,...]=TTA_ANGLES):
    if not tta or len(angles) == 0:
        x = preprocess_pil(img)[None, ...]                        # [1,3,H,W]
        heads = run_onnx(x)                                       # 4 x [1,10]
        logits = [h[0] for h in heads]                            # 4 x [10]
    else:
        xs = [preprocess_pil(img.rotate(a, resample=Image.BICUBIC)) for a in angles]
        x = np.stack(xs, axis=0)                                  # [K,3,H,W]
        heads = run_onnx(x)                                       # 4 x [K,10]
        logits = [h.mean(axis=0) for h in heads]                  # avg over K -> [10]
    digits = [int(l.argmax().item()) for l in logits]
    probs  = [float(softmax(l)[l.argmax()]) for l in logits]
    return "".join(map(str, digits)), probs

@app.post("/v1/predict")
async def predict(file: UploadFile = File(...), tta: bool = Query(False)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pred, probs = predict_pil(img, tta=tta)
    return {"prediction": pred, "per_digit_confidence": probs}

@app.post("/v1/predict/b64")
def predict_b64(batch: B64Batch):
    res = []
    for item in batch.items:
        img = Image.open(io.BytesIO(base64.b64decode(item.image_b64))).convert("RGB")
        pred, probs = predict_pil(img, tta=bool(item.tta))
        res.append({"prediction": pred, "per_digit_confidence": probs})
    return {"results": res}


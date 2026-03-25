"""
PostureAI Backend — FastAPI Server
==================================
Full-stack ML pipeline:
  Receive keypoint features → Feature validation → 
  LSTM/MLP classification → Return prediction

Endpoints:
  GET  /health       → Health check
  GET  /model/info   → Model metadata
  POST /predict      → Classify posture from feature window
  POST /data/submit  → Receive training data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import os
import time

from models.mlp_classifier import PostureMLP
from models.lstm_classifier import PostureLSTM
from models.feature_eng import FeatureValidator, FEATURE_NAMES, POSTURE_LABELS

app = FastAPI(
    title="PostureAI API",
    description="Real-time posture classification using temporal deep learning",
    version="2.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ───────────────────────────
class PredictRequest(BaseModel):
    features: List[List[float]]         # Sliding window: (seq_len, num_features)
    calibration: Optional[dict] = None  # User calibration baseline
    timestamp: Optional[int] = None

class PredictResponse(BaseModel):
    classIndex: int
    label: str
    confidence: float
    probabilities: List[float]
    score: int
    issues: List[str]
    model: str
    latency_ms: float

class DataSubmission(BaseModel):
    samples: List[dict]     # [{features: [...], label: int}, ...]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float


# ─── Global State ──────────────────────────────────────
START_TIME = time.time()
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model instances
mlp_model: Optional[PostureMLP] = None
lstm_model: Optional[PostureLSTM] = None
active_model_type = "rule-based"  # 'mlp', 'lstm', or 'rule-based'
feature_validator = FeatureValidator()


def load_models():
    """
    Try to load trained models from disk.
    Falls back to rule-based classification if no models found.
    """
    global mlp_model, lstm_model, active_model_type

    # Try LSTM first (preferred)
    lstm_path = os.path.join(MODEL_DIR, 'posture_lstm.pth')
    if os.path.exists(lstm_path):
        try:
            lstm_model = PostureLSTM()
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
            lstm_model.to(DEVICE)
            lstm_model.eval()
            active_model_type = "lstm"
            print(f"✅ LSTM model loaded from {lstm_path}")
            return
        except Exception as e:
            print(f"⚠️ Failed to load LSTM: {e}")

    # Try MLP as fallback
    mlp_path = os.path.join(MODEL_DIR, 'posture_mlp.pth')
    if os.path.exists(mlp_path):
        try:
            mlp_model = PostureMLP()
            mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
            mlp_model.to(DEVICE)
            mlp_model.eval()
            active_model_type = "mlp"
            print(f"✅ MLP model loaded from {mlp_path}")
            return
        except Exception as e:
            print(f"⚠️ Failed to load MLP: {e}")

    print("ℹ️  No trained models found. Using rule-based classification.")
    print(f"   Train models and place them in: {MODEL_DIR}")
    active_model_type = "rule-based"


@app.on_event("startup")
async def startup():
    load_models()


# ─── Endpoints ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=active_model_type != "rule-based",
        model_type=active_model_type,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/model/info")
async def model_info():
    info = {
        "active_model": active_model_type,
        "device": str(DEVICE),
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "num_classes": len(POSTURE_LABELS),
        "class_labels": POSTURE_LABELS,
        "models_available": {
            "lstm": lstm_model is not None,
            "mlp": mlp_model is not None,
        },
    }
    if lstm_model:
        info["lstm_params"] = sum(p.numel() for p in lstm_model.parameters())
    if mlp_model:
        info["mlp_params"] = sum(p.numel() for p in mlp_model.parameters())
    return info


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    start = time.time()

    # Validate input
    features = np.array(req.features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (seq_len, {len(FEATURE_NAMES)}), got {features.shape}"
        )

    # Validate feature ranges
    issues = feature_validator.validate(features[-1])  # Validate last frame

    # ── Classify ──
    if active_model_type == "lstm" and lstm_model is not None:
        result = _predict_lstm(features)
    elif active_model_type == "mlp" and mlp_model is not None:
        result = _predict_mlp(features[-1])   # MLP uses single frame
    else:
        result = _predict_rules(features)

    result["issues"] = issues
    result["model"] = active_model_type
    result["latency_ms"] = round((time.time() - start) * 1000, 2)

    return PredictResponse(**result)


def _predict_lstm(features: np.ndarray) -> dict:
    """Classify using LSTM (temporal model)."""
    # Pad/truncate to expected window size
    window_size = 30
    if features.shape[0] < window_size:
        padding = np.zeros((window_size - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.vstack([padding, features])
    else:
        features = features[-window_size:]

    # Run inference
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits = lstm_model(x)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    class_idx = int(np.argmax(probs))
    return {
        "classIndex": class_idx,
        "label": POSTURE_LABELS[class_idx],
        "confidence": float(probs[class_idx]),
        "probabilities": probs.tolist(),
        "score": _compute_score(class_idx, float(probs[class_idx])),
    }


def _predict_mlp(features: np.ndarray) -> dict:
    """Classify using MLP (single frame)."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits = mlp_model(x)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    class_idx = int(np.argmax(probs))
    return {
        "classIndex": class_idx,
        "label": POSTURE_LABELS[class_idx],
        "confidence": float(probs[class_idx]),
        "probabilities": probs.tolist(),
        "score": _compute_score(class_idx, float(probs[class_idx])),
    }


def _predict_rules(features: np.ndarray) -> dict:
    """Rule-based classification (when no ML model available)."""
    avg = features.mean(axis=0)
    scores = np.zeros(len(POSTURE_LABELS))
    scores[0] = 0.5  # Base score for good posture

    # Feature indices match FEATURE_NAMES
    shoulder = avg[0]
    neck = avg[1]
    torso = avg[2]
    head_offset = avg[4]
    symmetry = avg[7]
    ear = avg[9]

    if neck > 20:
        scores[1] += min(1.0, (neck - 20) / 20) * 0.5
        scores[0] -= 0.3
    if torso > 18:
        scores[1] += min(1.0, (torso - 18) / 15) * 0.4
        scores[0] -= 0.25
    if head_offset < -0.3:
        scores[2] += 0.4
        scores[0] -= 0.2
    elif head_offset > 0.3:
        scores[3] += 0.4
        scores[0] -= 0.2
    if ear > 25:
        scores[4] += min(1.0, (ear - 25) / 20) * 0.5
        scores[0] -= 0.3
    if shoulder > 12:
        s = min(1.0, (shoulder - 12) / 15) * 0.3
        if head_offset < 0:
            scores[2] += s
        else:
            scores[3] += s
        scores[0] -= 0.15

    # Softmax
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()
    class_idx = int(np.argmax(probs))

    return {
        "classIndex": class_idx,
        "label": POSTURE_LABELS[class_idx],
        "confidence": float(probs[class_idx]),
        "probabilities": probs.tolist(),
        "score": _compute_score(class_idx, float(probs[class_idx])),
    }


def _compute_score(class_idx: int, confidence: float) -> int:
    if class_idx == 0:
        return max(65, int(65 + confidence * 35))
    else:
        return max(0, int(65 - confidence * 65))


@app.post("/data/submit")
async def submit_data(data: DataSubmission):
    """Receive labeled training data from the frontend."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, f"samples_{int(time.time())}.npy")
    samples = []
    for s in data.samples:
        samples.append({
            "features": s.get("features", []),
            "label": s.get("label", 0),
        })

    np.save(filepath, samples, allow_pickle=True)
    return {"status": "saved", "count": len(samples), "path": filepath}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

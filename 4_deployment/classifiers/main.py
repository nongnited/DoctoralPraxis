"""
RAGShield — Classifier FastAPI Service
Runs on each shield container (ports 8101–8110).
Loads a trained model and serves /classify endpoint.

Environment variables:
  MODEL_NAME:   model identifier (e.g. naive_bayes, bert_base)
  MODEL_PATH:   path to models/ directory
  PORT:         port to serve on (8101–8110)

Endpoints:
  GET  /health        → {"status": "ok", "model": MODEL_NAME}
  POST /classify      → {"label": 0|1, "confidence": float, "latency_ms": float}
  GET  /model         → model metadata
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "naive_bayes")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models"))

app = FastAPI(title=f"RAGShield Classifier — {MODEL_NAME}")

# ── Global model state ────────────────────────────────────────────────────
_model   = None
_scaler  = None
_is_bert = False
_tokenizer = None
_config  = {}


def load_model():
    global _model, _scaler, _is_bert, _tokenizer, _config

    model_dir = MODEL_PATH / MODEL_NAME
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            _config = json.load(f)

    # BERT / HuggingFace model
    if (model_dir / "config.json").exists() and \
       any(model_dir.glob("*.safetensors")):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _is_bert = True
        log.info(f"Loading HuggingFace model from {model_dir}")
        _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        _model.eval()

        # Detect device
        if torch.cuda.is_available():
            _model = _model.cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _model = _model.to("mps")

        log.info(f"BERT model loaded: {MODEL_NAME}")

    else:
        # Classical sklearn/joblib model
        _is_bert = False
        _model = joblib.load(model_dir / "model.joblib")

        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            log.info(f"Scaler loaded for {MODEL_NAME}")

        log.info(f"Classical model loaded: {MODEL_NAME}")


@app.on_event("startup")
async def startup():
    load_model()


class ClassifyRequest(BaseModel):
    text: str
    feature: Optional[str] = "tfidf"  # tfidf | minilm | mpnet


class ClassifyResponse(BaseModel):
    label:         int
    confidence:    float
    latency_ms:    float
    model:         str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model is not None}


@app.get("/model")
def model_info():
    return {"model": MODEL_NAME, "is_bert": _is_bert, "config": _config}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    if _is_bert:
        label, confidence = _classify_bert(req.text)
    else:
        label, confidence = _classify_classical(req.text, req.feature)

    latency_ms = (time.perf_counter() - t0) * 1000

    return ClassifyResponse(
        label=label,
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
        model=MODEL_NAME,
    )


def _classify_bert(text: str):
    import torch
    device = next(_model.parameters()).device
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    label      = int(np.argmax(probs))
    confidence = float(probs[label])
    return label, confidence


def _classify_classical(text: str, feature: str):
    """Classify using classical model with TF-IDF vectorization."""
    # For deployment the vault gateway sends pre-vectorized features
    # This endpoint accepts raw text and vectorizes using the shared TF-IDF
    tfidf_path = Path(os.getenv("TFIDF_PATH", "/models/tfidf_vectorizer.joblib"))

    if not hasattr(_classify_classical, "_vectorizer"):
        if tfidf_path.exists():
            _classify_classical._vectorizer = joblib.load(tfidf_path)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"TF-IDF vectorizer not found at {tfidf_path}"
            )

    X = _classify_classical._vectorizer.transform([text])

    if _scaler is not None:
        X = _scaler.transform(X)

    if hasattr(_model, "predict_proba"):
        probs      = _model.predict_proba(X)[0]
        label      = int(np.argmax(probs))
        confidence = float(probs[label])
    else:
        label = int(_model.predict(X)[0])
        try:
            score      = _model.decision_function(X)[0]
            confidence = float(1 / (1 + np.exp(-abs(score))))
        except Exception:
            confidence = 1.0

    return label, confidence

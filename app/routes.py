"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
import io
import numpy as np

from .inference import run_inference
from .utils import validate_waveform_shape

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")

    raw = await file.read()
    try:
        data = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .npy file: {exc}") from exc

    try:
        waveform, _ = validate_waveform_shape(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = run_inference(waveform)

    return {
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "threshold_used": result.get("threshold_used"),
        "alert": result.get("alert"),
        "decision_reason": result.get("decision_reason"),
        "signal_energy": result.get("signal_energy"),
        "signal_variance": result.get("signal_variance"),
        "p_arrival_sec": result.get("p_arrival_sec"),
        "s_arrival_sec": result.get("s_arrival_sec"),
        "s_p_gap_sec": result.get("s_p_gap_sec"),
        "magnitude": result.get("magnitude"),
        "location_status": result.get("location_status", "experimental"),
    }

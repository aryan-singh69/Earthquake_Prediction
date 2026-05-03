"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router
from .inference import get_predictor

app = FastAPI(title="Earthquake Prediction API")
app.include_router(router)


@app.on_event("startup")
def startup_load_model():
	get_predictor()

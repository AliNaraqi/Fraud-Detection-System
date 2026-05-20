"""
api/main.py
FastAPI fraud detection service.

Endpoints:
  GET  /health          → liveness check
  GET  /model/info      → model metadata
  POST /predict         → single transaction
  POST /predict/batch   → batch scoring (up to 1000 txns)

Run:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.schemas import (
    BatchPredictionResponse,
    BatchPredictionResult,
    BatchTransactionRequest,
    FraudPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    TransactionRequest,
)
from models.predictor import _load_artefact, predict_batch, predict_single


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model…")
    try:
        _load_artefact()
        logger.success("Model loaded and ready.")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
    yield
    logger.info("API shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using XGBoost + SMOTE. "
        "Trained on synthetic transaction data with engineered features "
        "for velocity, geolocation, and behavioural patterns."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe — returns model load status."""
    loaded = config.MODEL_ARTIFACT_PATH.exists()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_path=str(config.MODEL_ARTIFACT_PATH),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Return model metadata: features, threshold, artefact path."""
    try:
        artefact = _load_artefact()
        return ModelInfoResponse(
            model_type="XGBoost + SMOTE",
            features=artefact["features"],
            decision_threshold=artefact["threshold"],
            artifact_path=str(config.MODEL_ARTIFACT_PATH),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python models/train.py first.",
        )


@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Score a single transaction",
)
def predict(transaction: TransactionRequest):
    """
    Predict fraud probability for a single transaction.

    Returns:
    - **fraud_probability**: 0–1 probability score
    - **is_fraud**: boolean decision at tuned threshold
    - **risk_tier**: low / medium / high / critical
    """
    try:
        result = predict_single(transaction.model_dump())
        return FraudPredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Score a batch of transactions (max 1000)",
)
def predict_batch_endpoint(request: BatchTransactionRequest):
    """
    Score multiple transactions in a single call.
    More efficient than repeated single-transaction calls.
    """
    try:
        rows  = [t.model_dump() for t in request.transactions]
        df    = pd.DataFrame(rows)
        result = predict_batch(df, log=True)

        predictions = [
            BatchPredictionResult(
                fraud_probability=float(row["fraud_probability"]),
                is_fraud=bool(row["is_fraud_predicted"]),
                risk_tier=str(row["risk_tier"]),
            )
            for _, row in result.iterrows()
        ]

        fraud_count = int(result["is_fraud_predicted"].sum())
        return BatchPredictionResponse(
            count=len(predictions),
            fraud_count=fraud_count,
            fraud_rate=round(fraud_count / len(predictions), 4),
            predictions=predictions,
            predicted_at=datetime.utcnow().isoformat(),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )

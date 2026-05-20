"""
models/predictor.py
Inference wrapper: load artefact → predict → log to predictions store.
Used by both the FastAPI endpoint and the monitoring pipeline.
"""
from __future__ import annotations

import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@lru_cache(maxsize=1)
def _load_artefact() -> dict:
    """Load and cache the model artefact bundle from disk."""
    if not config.MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {config.MODEL_ARTIFACT_PATH}. "
            "Run `python models/train.py` first."
        )
    artefact = joblib.load(config.MODEL_ARTIFACT_PATH)
    logger.info(f"Model loaded from {config.MODEL_ARTIFACT_PATH}")
    return artefact


def reload_model():
    """Clear the cache and reload the model (call after retraining)."""
    _load_artefact.cache_clear()
    _load_artefact()


def predict_batch(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """
    Run fraud predictions on a DataFrame with FEATURE_COLUMNS.
    Returns input df enriched with:  fraud_probability, is_fraud_predicted, risk_tier
    """
    artefact  = _load_artefact()
    model     = artefact["model"]
    scaler    = artefact["scaler"]
    threshold = artefact["threshold"]

    X       = df[config.FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X)
    probs   = model.predict_proba(X_scaled)[:, 1]
    preds   = (probs >= threshold).astype(int)

    result  = df.copy()
    result["fraud_probability"]   = probs.round(6)
    result["is_fraud_predicted"]  = preds
    result["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    ).astype(str)
    result["predicted_at"] = datetime.utcnow().isoformat()

    if log:
        _append_predictions_log(result)

    return result


def predict_single(features: dict) -> dict:
    """
    Predict fraud for a single transaction dict.
    Returns a structured response dict.
    """
    df      = pd.DataFrame([features])
    result  = predict_batch(df, log=True)
    row     = result.iloc[0]

    return {
        "fraud_probability":  float(row["fraud_probability"]),
        "is_fraud":           bool(row["is_fraud_predicted"]),
        "risk_tier":          row["risk_tier"],
        "decision_threshold": float(_load_artefact()["threshold"]),
        "predicted_at":       row["predicted_at"],
    }


def _append_predictions_log(df: pd.DataFrame):
    """Append predictions to the persistent Parquet log for drift monitoring."""
    cols = config.FEATURE_COLUMNS + [
        "fraud_probability", "is_fraud_predicted", "risk_tier", "predicted_at"
    ]
    # Only keep columns that exist
    keep = [c for c in cols if c in df.columns]
    log_df = df[keep].copy()

    if config.PREDICTIONS_LOG.exists():
        existing = pd.read_parquet(config.PREDICTIONS_LOG)
        combined = pd.concat([existing, log_df], ignore_index=True)
    else:
        combined = log_df

    combined.to_parquet(config.PREDICTIONS_LOG, index=False)


def get_predictions_log() -> pd.DataFrame:
    """Load the full predictions log (used by monitoring)."""
    if not config.PREDICTIONS_LOG.exists():
        return pd.DataFrame(columns=config.FEATURE_COLUMNS + ["fraud_probability"])
    return pd.read_parquet(config.PREDICTIONS_LOG)

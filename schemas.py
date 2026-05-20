"""
api/schemas.py
Pydantic v2 request and response schemas for the fraud detection API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Input payload for a single transaction fraud check."""

    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    merchant_category: int = Field(..., ge=0, le=7,
        description="Encoded merchant category (0-7)")
    country_code: int = Field(..., ge=0, le=9,
        description="Encoded country (0-9)")
    distance_from_home: float = Field(..., ge=0, description="km from home location")
    transactions_last_1h: int = Field(..., ge=0)
    transactions_last_24h: int = Field(..., ge=0)
    avg_amount_7d: float = Field(..., gt=0, description="7-day average transaction amount")
    is_international: int = Field(..., ge=0, le=1)
    is_new_device: int = Field(..., ge=0, le=1)
    is_weekend: int = Field(..., ge=0, le=1)
    velocity_ratio: float = Field(..., ge=0, description="txn_1h / (txn_24h/24)")
    amount_zscore: float = Field(..., description="z-score of amount vs 7d history")

    model_config = {"json_schema_extra": {
        "example": {
            "amount": 342.50,
            "hour_of_day": 2,
            "day_of_week": 6,
            "merchant_category": 3,
            "country_code": 5,
            "distance_from_home": 2400.0,
            "transactions_last_1h": 4,
            "transactions_last_24h": 18,
            "avg_amount_7d": 85.00,
            "is_international": 1,
            "is_new_device": 1,
            "is_weekend": 1,
            "velocity_ratio": 5.3,
            "amount_zscore": 3.02,
        }
    }}


class FraudPredictionResponse(BaseModel):
    """Single-transaction fraud prediction response."""
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_tier: Literal["low", "medium", "high", "critical"]
    decision_threshold: float
    predicted_at: str


class BatchTransactionRequest(BaseModel):
    """Batch scoring request."""
    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchPredictionResult(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_tier: str


class BatchPredictionResponse(BaseModel):
    count: int
    fraud_count: int
    fraud_rate: float
    predictions: list[BatchPredictionResult]
    predicted_at: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_path: str
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: str
    features: list[str]
    decision_threshold: float
    artifact_path: str

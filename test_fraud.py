"""
tests/test_fraud.py
Unit tests for data generation, model training helpers, and API schemas.

Run:  pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.generate import _make_transactions


# ── Data generation ───────────────────────────────────────────────────────────

def test_fraud_rate_approx():
    df = _make_transactions(5000, fraud_rate=0.015, seed=42)
    actual = df["is_fraud"].mean()
    assert 0.005 < actual < 0.04, f"Unexpected fraud rate: {actual:.3%}"


def test_feature_columns_present():
    df = _make_transactions(200, fraud_rate=0.015, seed=42)
    for col in config.FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature: {col}"


def test_no_nulls_in_features():
    df = _make_transactions(500, fraud_rate=0.02, seed=42)
    null_cols = df[config.FEATURE_COLUMNS].isnull().sum()
    assert null_cols.sum() == 0, f"Null values found:\n{null_cols[null_cols > 0]}"


def test_fraud_has_higher_amounts():
    df = _make_transactions(5000, fraud_rate=0.05, seed=42)
    fraud_mean  = df[df.is_fraud == 1]["amount"].mean()
    legit_mean  = df[df.is_fraud == 0]["amount"].mean()
    assert fraud_mean > legit_mean, "Fraud amounts should be higher on average"


def test_drift_shifts_distribution():
    clean = _make_transactions(2000, fraud_rate=0.02, seed=1, drift=False)
    drift = _make_transactions(2000, fraud_rate=0.02, seed=1, drift=True)
    assert drift["amount"].mean() > clean["amount"].mean() * 1.1


def test_derived_features_in_bounds():
    df = _make_transactions(1000, fraud_rate=0.02, seed=42)
    assert df["velocity_ratio"].ge(0).all(), "velocity_ratio must be >= 0"
    assert (df["amount_zscore"].abs() <= 10).all(), "amount_zscore clipped to [-10, 10]"


def test_target_is_binary():
    df = _make_transactions(500, fraud_rate=0.02, seed=42)
    assert set(df[config.TARGET_COLUMN].unique()).issubset({0, 1})


# ── Training helpers ──────────────────────────────────────────────────────────

def test_optimal_threshold_between_0_and_1():
    from models.train import _optimal_threshold
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    thresh = _optimal_threshold(y_true, y_prob)
    assert 0.0 < thresh < 1.0


# ── API schemas ───────────────────────────────────────────────────────────────

def test_transaction_request_validation():
    from api.schemas import TransactionRequest
    txn = TransactionRequest(
        amount=100.0,
        hour_of_day=14,
        day_of_week=3,
        merchant_category=2,
        country_code=0,
        distance_from_home=5.0,
        transactions_last_1h=1,
        transactions_last_24h=7,
        avg_amount_7d=95.0,
        is_international=0,
        is_new_device=0,
        is_weekend=0,
        velocity_ratio=3.4,
        amount_zscore=0.05,
    )
    assert txn.amount == 100.0
    assert txn.is_international == 0


def test_transaction_rejects_negative_amount():
    from api.schemas import TransactionRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransactionRequest(
            amount=-50,         # invalid
            hour_of_day=10,
            day_of_week=1,
            merchant_category=0,
            country_code=0,
            distance_from_home=10,
            transactions_last_1h=1,
            transactions_last_24h=5,
            avg_amount_7d=80,
            is_international=0,
            is_new_device=0,
            is_weekend=0,
            velocity_ratio=1.2,
            amount_zscore=-0.3,
        )


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_paths_exist():
    assert config.DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.LOGS_DIR.exists()

def test_feature_column_count():
    assert len(config.FEATURE_COLUMNS) == 14

def test_fraud_rate_is_low():
    assert config.FRAUD_RATE < 0.05, "Fraud rate should be realistically low (<5%)"

def test_smote_sampling_below_one():
    assert 0 < config.SMOTE_SAMPLING_STRATEGY < 1

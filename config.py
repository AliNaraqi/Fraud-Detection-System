"""
config.py  —  Central configuration for the fraud detection system.
All tuneable knobs live here. Override via .env or environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
MODELS_DIR   = BASE_DIR / "models"
REPORTS_DIR  = BASE_DIR / "reports"
LOGS_DIR     = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
N_SAMPLES           = 50_000
FRAUD_RATE          = 0.015          # 1.5 %  →  severe class imbalance
RANDOM_SEED         = 42
TEST_SIZE           = 0.20
REFERENCE_SIZE      = 5_000          # rows kept as Evidently reference set

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category",   # ordinal-encoded
    "country_code",        # ordinal-encoded
    "distance_from_home",
    "transactions_last_1h",
    "transactions_last_24h",
    "avg_amount_7d",
    "is_international",
    "is_new_device",
    "is_weekend",
    "velocity_ratio",      # txn count / 7d average
    "amount_zscore",       # z-score vs customer history
]
TARGET_COLUMN = "is_fraud"

# ── SMOTE ─────────────────────────────────────────────────────────────────────
SMOTE_SAMPLING_STRATEGY = 0.20   # after SMOTE: 20% minority / 80% majority
SMOTE_K_NEIGHBORS       = 5

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":       500,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "scale_pos_weight":   1,       # handled by SMOTE, so keep at 1
    "eval_metric":        "aucpr",
    "random_state":       RANDOM_SEED,
    "n_jobs":             -1,
}

# Decision threshold — tuned for high recall on fraud
# Lower  →  more fraud caught, more false positives
# Higher →  fewer alerts, more missed fraud
DECISION_THRESHOLD = 0.35

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME = "fraud-detection"
REGISTERED_MODEL_NAME  = "fraud-detector"

# ── Evidently drift thresholds ────────────────────────────────────────────────
DRIFT_SHARE_THRESHOLD  = 0.30    # alert if > 30 % of features drift
PSI_THRESHOLD          = 0.10    # Population Stability Index

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Model artefact path ───────────────────────────────────────────────────────
MODEL_ARTIFACT_PATH = MODELS_DIR / "fraud_model.joblib"
REFERENCE_DATA_PATH = DATA_DIR   / "reference.parquet"
PREDICTIONS_LOG     = LOGS_DIR   / "predictions.parquet"

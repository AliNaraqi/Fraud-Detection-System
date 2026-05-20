"""
data/generate.py
Generates a realistic synthetic credit-card fraud dataset.

Key properties:
  - Severe class imbalance  (~1.5 % fraud)
  - Correlated features     (fraud transactions share statistical patterns)
  - Injected production drift for monitoring demos

Run standalone:  python data/generate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

rng = np.random.default_rng(config.RANDOM_SEED)

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail",
    "travel", "entertainment", "atm", "pharmacy",
]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "MX", "CN", "BR", "IN", "AU"]
MERCHANT_ENC = {v: i for i, v in enumerate(MERCHANT_CATEGORIES)}
COUNTRY_ENC  = {v: i for i, v in enumerate(COUNTRIES)}

# Fraud transactions skew toward late-night hours (must sum to 1)
_FRAUD_HOUR_P = np.array([
    0.06, 0.06, 0.07, 0.07, 0.06, 0.04,
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.04, 0.04,
    0.05, 0.05, 0.06, 0.06, 0.07, 0.07,
])
_FRAUD_HOUR_P = _FRAUD_HOUR_P / _FRAUD_HOUR_P.sum()


def _make_transactions(n: int, fraud_rate: float, seed: int, drift: bool = False) -> pd.DataFrame:
    rng_local = np.random.default_rng(seed)
    n_fraud   = int(n * fraud_rate)
    n_legit   = n - n_fraud

    # ── Legitimate transactions ───────────────────────────────────────────────
    legit = {
        "amount":               rng_local.lognormal(mean=3.5, sigma=1.2, size=n_legit).clip(0.5, 5000),
        "hour_of_day":          rng_local.integers(0, 24, size=n_legit),
        "day_of_week":          rng_local.integers(0, 7,  size=n_legit),
        "merchant_category":    rng_local.choice(list(MERCHANT_ENC.values()), size=n_legit,
                                                 p=[0.20,0.15,0.18,0.15,0.08,0.10,0.07,0.07]),
        "country_code":         rng_local.choice(list(COUNTRY_ENC.values()), size=n_legit,
                                                 p=[0.45,0.10,0.10,0.07,0.07,0.05,0.05,0.04,0.04,0.03]),
        "distance_from_home":   rng_local.exponential(scale=20, size=n_legit).clip(0, 10000),
        "transactions_last_1h": rng_local.poisson(lam=1.2, size=n_legit).clip(0, 20),
        "transactions_last_24h":rng_local.poisson(lam=8,   size=n_legit).clip(0, 100),
        "avg_amount_7d":        rng_local.lognormal(mean=3.4, sigma=0.8, size=n_legit).clip(1, 2000),
        "is_international":     rng_local.choice([0, 1], size=n_legit, p=[0.85, 0.15]),
        "is_new_device":        rng_local.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "is_weekend":           rng_local.choice([0, 1], size=n_legit, p=[0.71, 0.29]),
        "is_fraud":             np.zeros(n_legit, dtype=int),
    }

    # ── Fraudulent transactions  (different statistical profile) ─────────────
    fraud = {
        "amount":               rng_local.lognormal(mean=5.0, sigma=1.5, size=n_fraud).clip(10, 5000),
        "hour_of_day":          rng_local.choice(np.arange(24), size=n_fraud, p=_FRAUD_HOUR_P),
        "day_of_week":          rng_local.integers(0, 7, size=n_fraud),
        "merchant_category":    rng_local.choice(list(MERCHANT_ENC.values()), size=n_fraud,
                                                 p=[0.05,0.05,0.05,0.35,0.10,0.10,0.25,0.05]),
        "country_code":         rng_local.choice(list(COUNTRY_ENC.values()), size=n_fraud,
                                                 p=[0.15,0.05,0.08,0.08,0.08,0.15,0.15,0.10,0.08,0.08]),
        "distance_from_home":   rng_local.exponential(scale=500, size=n_fraud).clip(0, 10000),
        "transactions_last_1h": rng_local.poisson(lam=5,   size=n_fraud).clip(0, 20),
        "transactions_last_24h":rng_local.poisson(lam=20,  size=n_fraud).clip(0, 100),
        "avg_amount_7d":        rng_local.lognormal(mean=3.0, sigma=0.8, size=n_fraud).clip(1, 2000),
        "is_international":     rng_local.choice([0, 1], size=n_fraud, p=[0.40, 0.60]),
        "is_new_device":        rng_local.choice([0, 1], size=n_fraud, p=[0.50, 0.50]),
        "is_weekend":           rng_local.choice([0, 1], size=n_fraud, p=[0.60, 0.40]),
        "is_fraud":             np.ones(n_fraud, dtype=int),
    }

    df = pd.concat(
        [pd.DataFrame(legit), pd.DataFrame(fraud)],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Derived features ──────────────────────────────────────────────────────
    df["velocity_ratio"] = (
        df["transactions_last_1h"] /
        (df["transactions_last_24h"] / 24.0).clip(lower=0.01)
    ).round(3)

    df["amount_zscore"] = (
        (df["amount"] - df["avg_amount_7d"]) /
        df["avg_amount_7d"].clip(lower=1.0)
    ).round(4)

    # Drift: shift amount distribution and increase international txns
    if drift:
        df["amount"]          *= rng_local.uniform(1.3, 2.0, size=len(df))
        df["is_international"] = rng_local.choice([0,1], size=len(df), p=[0.50, 0.50])
        df["transactions_last_1h"] = (df["transactions_last_1h"] * 1.5).clip(0, 20).astype(int)

    df["amount"]        = df["amount"].round(2)
    df["amount_zscore"] = df["amount_zscore"].clip(-10, 10)

    return df


def generate_all() -> dict[str, Path]:
    """Generate and persist train / reference / production datasets."""
    logger.info(f"Generating {config.N_SAMPLES:,} transactions "
                f"(fraud_rate={config.FRAUD_RATE:.1%})…")

    train_df = _make_transactions(
        config.N_SAMPLES, config.FRAUD_RATE, seed=config.RANDOM_SEED
    )
    ref_df = train_df.sample(
        n=config.REFERENCE_SIZE, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    prod_df = _make_transactions(
        n=5_000, fraud_rate=config.FRAUD_RATE,
        seed=config.RANDOM_SEED + 99, drift=True,
    )
    prod_clean_df = _make_transactions(
        n=5_000, fraud_rate=config.FRAUD_RATE,
        seed=config.RANDOM_SEED + 200, drift=False,
    )

    paths = {}
    for name, df in [
        ("train", train_df),
        ("reference", ref_df),
        ("production_drift", prod_df),
        ("production_clean", prod_clean_df),
    ]:
        p = config.DATA_DIR / f"{name}.parquet"
        df.to_parquet(p, index=False)
        fraud_count = df["is_fraud"].sum()
        logger.info(f"  {name}: {len(df):,} rows  |  fraud: {fraud_count} ({fraud_count/len(df):.2%})")
        paths[name] = p

    return paths


if __name__ == "__main__":
    generate_all()
    logger.success("Datasets generated.")

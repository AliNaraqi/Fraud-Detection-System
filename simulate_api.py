"""
scripts/simulate_api.py
Simulate a stream of transactions hitting the live FastAPI endpoint.
Useful for populating the prediction log for drift monitoring demos.

Usage:
  1. Start the API:    uvicorn api.main:app --reload --port 8000
  2. Run this script: python scripts/simulate_api.py

Options:
  --n       number of transactions to send  (default: 200)
  --batch   batch size per request          (default: 20)
  --url     API base URL                    (default: http://localhost:8000)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.generate import _make_transactions


def run_simulation(n: int = 200, batch: int = 20, url: str = "http://localhost:8000"):
    rng_sim = np.random.default_rng(777)
    df = _make_transactions(n, fraud_rate=config.FRAUD_RATE, seed=777, drift=True)

    logger.info(f"Sending {n} transactions to {url} in batches of {batch}…")
    total_fraud = 0

    with httpx.Client(timeout=30) as client:
        for start in range(0, n, batch):
            chunk = df.iloc[start:start + batch]
            payload = {"transactions": chunk[config.FEATURE_COLUMNS].to_dict(orient="records")}

            try:
                resp = client.post(f"{url}/predict/batch", json=payload)
                resp.raise_for_status()
                data = resp.json()
                total_fraud += data["fraud_count"]
                logger.info(
                    f"  Batch {start//batch + 1}: "
                    f"{data['count']} txns, "
                    f"{data['fraud_count']} fraud ({data['fraud_rate']:.1%})"
                )
            except httpx.ConnectError:
                logger.error(f"Could not connect to {url}. Is the API running?")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Request failed: {e}")

            time.sleep(0.1)

    logger.success(
        f"Done! Sent {n} transactions, flagged {total_fraud} as fraud "
        f"({total_fraud/n:.1%})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=200)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--url",   type=str, default="http://localhost:8000")
    args = parser.parse_args()
    run_simulation(args.n, args.batch, args.url)

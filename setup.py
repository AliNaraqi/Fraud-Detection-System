"""
scripts/setup.py
One-command bootstrap:
  1. Generate datasets (train + reference + production)
  2. Train XGBoost + SMOTE, register in MLflow
  3. Run first monitoring pass with Evidently
  4. Print next steps

Run:  python scripts/setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("Fraud Detection System — Setup")
    logger.info("=" * 60)

    # 1. Generate data
    logger.info("\n[1/3] Generating synthetic fraud datasets…")
    from data.generate import generate_all
    paths = generate_all()
    for name, p in paths.items():
        logger.info(f"  ✓ {name}: {p}")

    # 2. Train model
    logger.info("\n[2/3] Training XGBoost + SMOTE model…")
    from models.train import train
    run_id = train()
    logger.info(f"  ✓ Training complete — run_id: {run_id}")

    # 3. First monitoring run
    logger.info("\n[3/3] Running Evidently drift monitoring (demo drift data)…")
    from monitoring.monitor_pipeline import run_monitoring
    result = run_monitoring(use_drift_data=True)
    logger.info(f"  ✓ Drift detected:  {result['drift_detected']}")
    logger.info(f"  ✓ Drifted features: {result['drifted_features']}")
    logger.info(f"  ✓ Alerts:          {len(result['alerts'])}")

    # Done
    logger.info("\n" + "=" * 60)
    logger.info("Setup complete! Next steps:")
    logger.info("")
    logger.info("  Dashboard:   streamlit run dashboard/app.py")
    logger.info("  API server:  uvicorn api.main:app --reload --port 8000")
    logger.info("  API docs:    http://localhost:8000/docs")
    logger.info("  MLflow UI:   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("  Tests:       pytest tests/ -v")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
monitoring/monitor_pipeline.py
Orchestrated monitoring run:
  1. Load reference data (training distribution)
  2. Load recent production predictions
  3. Run Evidently drift reports
  4. Log results + fire alerts
  5. Trigger retraining if drift is severe

Can be run standalone or called from a scheduler.
Run:  python monitoring/monitor_pipeline.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from monitoring.drift_monitor import (
    run_data_drift_report,
    run_drift_test_suite,
    run_prediction_drift_report,
)
from models.predictor import get_predictions_log


MONITOR_LOG = config.LOGS_DIR / "monitoring_runs.jsonl"


def _load_reference() -> pd.DataFrame:
    if not config.REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found: {config.REFERENCE_DATA_PATH}\n"
            "Run `python scripts/setup.py` first."
        )
    return pd.read_parquet(config.REFERENCE_DATA_PATH)


def _load_production(use_drift_file: bool = False) -> pd.DataFrame:
    """
    Load production data.
    use_drift_file=True  →  load the pre-built drifted parquet (for demos)
    use_drift_file=False →  load the live predictions log from the API
    """
    if use_drift_file:
        p = config.DATA_DIR / "production_drift.parquet"
        if not p.exists():
            from data.generate import generate_all
            generate_all()
        return pd.read_parquet(p)

    log_df = get_predictions_log()
    if len(log_df) < 50:
        logger.warning(
            f"Only {len(log_df)} predictions in log — "
            "using pre-built drift file as fallback."
        )
        p = config.DATA_DIR / "production_drift.parquet"
        if p.exists():
            return pd.read_parquet(p)
    return log_df


def run_monitoring(use_drift_data: bool = True) -> dict:
    """
    Full monitoring run. Returns a summary dict.

    Parameters
    ----------
    use_drift_data : bool
        True  →  force-load the pre-built drifted dataset (demo mode)
        False →  use live prediction log from the API
    """
    run_id = str(uuid.uuid4())[:12]
    logger.info(f"Monitoring run started | id={run_id} | drift_demo={use_drift_data}")

    reference  = _load_reference()
    production = _load_production(use_drift_file=use_drift_data)

    logger.info(f"Reference:  {len(reference):,} rows")
    logger.info(f"Production: {len(production):,} rows")

    # ── Evidently reports ─────────────────────────────────────────────────────
    logger.info("Running data drift report…")
    drift_summary = run_data_drift_report(reference, production, run_id=run_id)

    logger.info("Running prediction drift report…")
    pred_drift = run_prediction_drift_report(reference, production)

    logger.info("Running drift test suite…")
    test_results = run_drift_test_suite(reference, production)

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts = []
    if drift_summary["dataset_drift_detected"]:
        share = drift_summary["drift_share"]
        level = "CRITICAL" if share > 0.5 else "WARNING"
        msg = (
            f"{level}: Data drift in {len(drift_summary['drifted_features'])} features "
            f"({share:.0%} drift share). "
            f"Drifted: {', '.join(drift_summary['drifted_features'][:5])}"
        )
        alerts.append({"level": level, "message": msg, "type": "data_drift"})
        logger.warning(f"🚨 {msg}")

    if pred_drift.get("prediction_drift_detected"):
        msg = (
            f"WARNING: Prediction distribution has shifted "
            f"(score={pred_drift.get('prediction_drift_score', 0):.4f})"
        )
        alerts.append({"level": "WARNING", "message": msg, "type": "prediction_drift"})
        logger.warning(f"⚠  {msg}")

    if test_results["failed"] > 0:
        msg = (
            f"WARNING: {test_results['failed']}/{test_results['total_tests']} "
            "drift tests failed."
        )
        alerts.append({"level": "WARNING", "message": msg, "type": "test_failure"})
        logger.warning(f"⚠  {msg}")

    if not alerts:
        logger.success("✓ All monitoring checks passed — no drift detected.")

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"monitor_{run_id}"):
        mlflow.set_tag("monitoring_type", "drift")
        mlflow.set_tag("run_id",          run_id)
        mlflow.set_tag("drift_detected",  str(drift_summary["dataset_drift_detected"]))
        mlflow.set_tag("alerts",          str(len(alerts)))

        mlflow.log_metric("drift_share",      drift_summary["drift_share"])
        mlflow.log_metric("n_drifted_features", len(drift_summary["drifted_features"]))
        mlflow.log_metric("test_pass_rate",   test_results["pass_rate"])
        mlflow.log_metric("n_alerts",         len(alerts))

        for report_key in ["html_report"]:
            path = drift_summary.get(report_key)
            if path and Path(path).exists():
                mlflow.log_artifact(path, "reports")

    # ── Persist run log ───────────────────────────────────────────────────────
    record = {
        "run_id":           run_id,
        "timestamp":        datetime.utcnow().isoformat(),
        "drift_detected":   drift_summary["dataset_drift_detected"],
        "drift_share":      drift_summary["drift_share"],
        "drifted_features": drift_summary["drifted_features"],
        "pred_drift":       pred_drift.get("prediction_drift_detected", False),
        "tests_passed":     test_results["passed"],
        "tests_total":      test_results["total_tests"],
        "alerts":           alerts,
        "feature_scores":   drift_summary.get("feature_drift_scores", {}),
    }
    with open(MONITOR_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Monitoring complete | alerts={len(alerts)} | run_id={run_id}")
    return record


def load_monitoring_history(n: int = 50) -> list[dict]:
    """Load recent monitoring runs from the JSONL log."""
    if not MONITOR_LOG.exists():
        return []
    lines = MONITOR_LOG.read_text().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()][-n:]


if __name__ == "__main__":
    result = run_monitoring(use_drift_data=True)
    print(json.dumps(result, indent=2, default=str))

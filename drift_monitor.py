"""
monitoring/drift_monitor.py
Evidently AI drift detection for the fraud model.

Compares the reference (training) distribution against recent production
predictions to detect feature drift, prediction drift, and data quality issues.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.test_preset import DataDriftTestPreset
from evidently.legacy.test_suite import TestSuite
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Column mapping ────────────────────────────────────────────────────────────

def _col_mapping() -> ColumnMapping:
    numerical = [
        "amount", "distance_from_home", "transactions_last_1h",
        "transactions_last_24h", "avg_amount_7d", "velocity_ratio",
        "amount_zscore",
    ]
    categorical = [
        "hour_of_day", "day_of_week", "merchant_category", "country_code",
        "is_international", "is_new_device", "is_weekend",
    ]
    return ColumnMapping(
        target=config.TARGET_COLUMN,
        prediction="is_fraud_predicted",
        numerical_features=numerical,
        categorical_features=categorical,
    )


# ── Report builders ───────────────────────────────────────────────────────────

def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_html: bool = True,
    run_id: Optional[str] = None,
) -> dict:
    """
    Full Evidently data drift report.
    Returns a summary dict; optionally saves HTML to reports/.
    """
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetMissingValuesMetric(),
    ])

    ref_feats = reference[config.FEATURE_COLUMNS]
    cur_feats = current[config.FEATURE_COLUMNS]

    report.run(
        reference_data=ref_feats,
        current_data=cur_feats,
        column_mapping=ColumnMapping(
            numerical_features=[
                "amount", "distance_from_home", "transactions_last_1h",
                "transactions_last_24h", "avg_amount_7d", "velocity_ratio",
                "amount_zscore",
            ],
            categorical_features=[
                "hour_of_day", "day_of_week", "merchant_category", "country_code",
                "is_international", "is_new_device", "is_weekend",
            ],
        ),
    )

    summary = _parse_drift_summary(report.as_dict())

    if save_html:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_id}" if run_id else ""
        path = config.REPORTS_DIR / f"data_drift{suffix}_{ts}.html"
        report.save_html(str(path))
        summary["html_report"] = str(path)
        logger.info(f"Drift report → {path}")

    return summary


def run_prediction_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_html: bool = True,
) -> dict:
    """Check if the model's fraud prediction distribution has shifted."""
    # Ensure both DataFrames have the prediction column
    if "fraud_probability" not in reference.columns:
        logger.warning("No fraud_probability in reference — skipping prediction drift.")
        return {"prediction_drift_detected": False}
    if "fraud_probability" not in current.columns:
        return {"prediction_drift_detected": False}

    report = Report(metrics=[
        ColumnDriftMetric(column_name="fraud_probability"),
        TargetDriftPreset(),
    ])

    report.run(
        reference_data=reference[["fraud_probability"]],
        current_data=current[["fraud_probability"]],
        column_mapping=ColumnMapping(prediction="fraud_probability"),
    )

    result = {"prediction_drift_detected": False, "prediction_drift_score": 0.0}
    try:
        for metric in report.as_dict().get("metrics", []):
            if metric.get("metric") == "ColumnDriftMetric":
                r = metric.get("result", {})
                result["prediction_drift_detected"] = r.get("drift_detected", False)
                result["prediction_drift_score"]    = r.get("drift_score", 0.0)
    except Exception as e:
        logger.warning(f"Could not parse prediction drift: {e}")

    if save_html:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = config.REPORTS_DIR / f"prediction_drift_{ts}.html"
        report.save_html(str(path))
        result["html_report"] = str(path)

    return result


def run_drift_test_suite(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_html: bool = True,
) -> dict:
    """
    Evidently TestSuite — binary pass/fail for data quality and drift.
    """
    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(
        reference_data=reference[config.FEATURE_COLUMNS],
        current_data=current[config.FEATURE_COLUMNS],
        column_mapping=ColumnMapping(
            numerical_features=[
                "amount", "distance_from_home", "transactions_last_1h",
                "transactions_last_24h", "avg_amount_7d", "velocity_ratio",
                "amount_zscore",
            ],
            categorical_features=[
                "hour_of_day", "day_of_week", "merchant_category",
                "country_code", "is_international", "is_new_device", "is_weekend",
            ],
        ),
    )

    suite_dict = suite.as_dict()
    total   = len(suite_dict.get("tests", []))
    passed  = sum(1 for t in suite_dict.get("tests", []) if t.get("status") == "SUCCESS")

    result = {
        "total_tests": total,
        "passed":      passed,
        "failed":      total - passed,
        "pass_rate":   round(passed / total, 3) if total else 0.0,
    }

    if save_html:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = config.REPORTS_DIR / f"test_suite_{ts}.html"
        suite.save_html(str(path))
        result["html_report"] = str(path)

    return result


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_drift_summary(report_dict: dict) -> dict:
    """Extract key metrics from the Evidently report JSON."""
    result = {
        "dataset_drift_detected": False,
        "drift_share":            0.0,
        "drifted_features":       [],
        "feature_drift_scores":   {},
        "missing_values_share":   0.0,
    }
    try:
        for metric in report_dict.get("metrics", []):
            m = metric.get("metric", "")
            r = metric.get("result", {})

            if m == "DatasetDriftMetric":
                result["dataset_drift_detected"] = r.get("dataset_drift", False)
                result["drift_share"]            = r.get("share_of_drifted_columns", 0.0)

            elif m == "DataDriftTable":
                for col, info in r.get("drift_by_columns", {}).items():
                    score   = info.get("drift_score", 0.0)
                    drifted = info.get("drift_detected", False)
                    result["feature_drift_scores"][col] = round(score, 4)
                    if drifted:
                        result["drifted_features"].append(col)

            elif m == "DatasetMissingValuesMetric":
                result["missing_values_share"] = (
                    r.get("current", {}).get("share_of_missing_values", 0.0)
                )
    except Exception as e:
        logger.warning(f"Drift parse error: {e}")

    return result

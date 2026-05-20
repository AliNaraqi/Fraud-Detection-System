"""
models/train.py
Training pipeline:
  1. Load data
  2. Train/test split (stratified)
  3. Apply SMOTE on training set only  (never on test!)
  4. Train XGBoost with early stopping
  5. Tune decision threshold for max F1
  6. Log everything to MLflow
  7. Register best model
  8. Save artefact to disk for API

Run:  python models/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.generate import generate_all


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_generate() -> pd.DataFrame:
    train_path = config.DATA_DIR / "train.parquet"
    if not train_path.exists():
        logger.info("No training data found — generating…")
        generate_all()
    return pd.read_parquet(train_path)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / np.where(
        (precision + recall) == 0, 1e-9, (precision + recall)
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


# ── Main training function ────────────────────────────────────────────────────

def train() -> str:
    """Full training pipeline. Returns MLflow run_id."""

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df    = _load_or_generate()
    X     = df[config.FEATURE_COLUMNS]
    y     = df[config.TARGET_COLUMN]

    logger.info(f"Dataset: {len(df):,} rows | fraud: {y.sum()} ({y.mean():.2%})")

    # ── Stratified split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    logger.info(f"Train fraud rate: {y_train.mean():.2%}")

    with mlflow.start_run(run_name="xgb_smote") as run:

        # ── Log dataset stats ─────────────────────────────────────────────────
        mlflow.log_params({
            "n_train":                len(X_train),
            "n_test":                 len(X_test),
            "train_fraud_rate":       round(float(y_train.mean()), 4),
            "smote_sampling_strategy": config.SMOTE_SAMPLING_STRATEGY,
            "smote_k_neighbors":       config.SMOTE_K_NEIGHBORS,
            **config.XGB_PARAMS,
        })
        mlflow.set_tag("model_type", "XGBoost + SMOTE")

        # ── SMOTE  (training set only!) ───────────────────────────────────────
        logger.info("Applying SMOTE…")
        smote = SMOTE(
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            k_neighbors=config.SMOTE_K_NEIGHBORS,
            random_state=config.RANDOM_SEED,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        logger.info(
            f"After SMOTE: {len(X_res):,} rows | "
            f"fraud: {y_res.sum()} ({y_res.mean():.2%})"
        )
        mlflow.log_params({
            "post_smote_samples":    len(X_res),
            "post_smote_fraud_rate": round(float(y_res.mean()), 4),
        })

        # ── Train XGBoost ─────────────────────────────────────────────────────
        logger.info("Training XGBoost…")
        scaler = StandardScaler()
        X_res_scaled  = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        # Eval set for early stopping
        eval_set = [(X_test_scaled, y_test)]
        xgb_params = {k: v for k, v in config.XGB_PARAMS.items()
                      if k != "use_label_encoder"}

        model = XGBClassifier(**xgb_params)
        model.fit(
            X_res_scaled, y_res,
            eval_set=eval_set,
            verbose=50,
        )

        # ── Evaluate at default threshold ─────────────────────────────────────
        y_prob  = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_default = (y_prob >= 0.5).astype(int)

        # ── Tune decision threshold ───────────────────────────────────────────
        optimal_thresh = _optimal_threshold(y_test.values, y_prob)
        y_pred_tuned   = (y_prob >= optimal_thresh).astype(int)
        logger.info(f"Optimal decision threshold: {optimal_thresh:.4f}")

        # ── Metrics ───────────────────────────────────────────────────────────
        roc_auc  = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        f1_tuned = f1_score(y_test, y_pred_tuned)
        f1_def   = f1_score(y_test, y_pred_default)

        cm = confusion_matrix(y_test, y_pred_tuned)
        tn, fp, fn, tp = cm.ravel()

        logger.info(f"ROC-AUC:     {roc_auc:.4f}")
        logger.info(f"Avg Prec:    {avg_prec:.4f}")
        logger.info(f"F1 (tuned):  {f1_tuned:.4f}  |  F1 (0.5): {f1_def:.4f}")
        logger.info(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        mlflow.log_metrics({
            "roc_auc":           roc_auc,
            "avg_precision":     avg_prec,
            "f1_default":        f1_def,
            "f1_tuned":          f1_tuned,
            "tp":                int(tp),
            "fp":                int(fp),
            "fn":                int(fn),
            "tn":                int(tn),
            "precision_tuned":   tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall_tuned":      tp / (tp + fn) if (tp + fn) > 0 else 0,
            "optimal_threshold": optimal_thresh,
        })

        report = classification_report(y_test, y_pred_tuned,
                                       target_names=["legitimate", "fraud"])
        mlflow.log_text(report, "classification_report.txt")
        logger.info(f"\n{report}")

        # ── Feature importances ───────────────────────────────────────────────
        fi = pd.DataFrame({
            "feature":    config.FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        mlflow.log_text(fi.to_string(index=False), "feature_importances.txt")
        logger.info(f"\nTop 5 features:\n{fi.head().to_string(index=False)}")

        # ── Save artefact bundle  (model + scaler + threshold) ────────────────
        artefact = {
            "model":     model,
            "scaler":    scaler,
            "threshold": optimal_thresh,
            "features":  config.FEATURE_COLUMNS,
        }
        joblib.dump(artefact, config.MODEL_ARTIFACT_PATH)
        mlflow.log_artifact(str(config.MODEL_ARTIFACT_PATH))

        # ── Log model to registry ─────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model",
            registered_model_name=config.REGISTERED_MODEL_NAME,
        )

        run_id = run.info.run_id

    logger.success(f"Training complete  |  run_id={run_id}")
    return run_id


if __name__ == "__main__":
    train()

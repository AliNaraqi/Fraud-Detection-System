# 🛡 Fraud Detection System

End-to-end credit card fraud detection: XGBoost + SMOTE for severe class imbalance, FastAPI for real-time scoring, Evidently AI for production drift monitoring, and MLflow for experiment tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)
![Evidently](https://img.shields.io/badge/Evidently-0.4-FF5C5C?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

| Component | Details |
|---|---|
| 🧬 Synthetic data | 50,000 transactions at 1.5% fraud rate — severe class imbalance |
| ⚖️ SMOTE | Oversampling on training set only — never leaks into test |
| 🌲 XGBoost | 500 estimators, early stopping, threshold tuning for max F1 |
| 📊 MLflow | All runs, metrics, artefacts, and model registry |
| 🚀 FastAPI | `/predict` (single) + `/predict/batch` (up to 1000) with Pydantic v2 |
| 🔍 Evidently | DataDrift + TargetDrift + TestSuite with HTML reports |
| 🖥 Dashboard | Streamlit: live predictor, drift charts, prediction log |
| 🧪 Tests | pytest unit tests for data, model, and API schemas |
| 📡 Simulator | Script to stream test transactions to the live API |

---

## 📁 Project Structure

```
fraud-detection/
├── config.py                        ← all thresholds + paths
├── .env.example
├── requirements.txt
│
├── data/
│   └── generate.py                  ← synthetic fraud dataset generator (with drift)
│
├── models/
│   ├── train.py                     ← SMOTE + XGBoost + threshold tuning + MLflow
│   └── predictor.py                 ← inference wrapper + prediction log
│
├── api/
│   ├── main.py                      ← FastAPI app (single + batch endpoints)
│   └── schemas.py                   ← Pydantic v2 request/response models
│
├── monitoring/
│   ├── drift_monitor.py             ← Evidently reports (data + prediction drift)
│   └── monitor_pipeline.py          ← end-to-end monitoring run + alert log
│
├── dashboard/
│   └── app.py                       ← Streamlit: overview, drift, predictor, log
│
├── scripts/
│   ├── setup.py                     ← one-command full setup
│   └── simulate_api.py              ← stream test transactions to the API
│
└── tests/
    └── test_fraud.py                ← pytest unit tests
```

---



## 🏗 System Architecture

```
                     ┌─────────────────────┐
                     │   data/generate.py  │
                     │  50k transactions   │
                     │  1.5% fraud rate    │
                     └──────────┬──────────┘
                                │
                    ┌───────────▼───────────┐
                    │     models/train.py   │
                    │                       │
                    │  StratifiedSplit 80/20 │
                    │  SMOTE on train only  │
                    │  XGBoost (500 trees)  │
                    │  Threshold tuning     │
                    │  MLflow tracking      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
   ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
   │  FastAPI         │  │  monitoring/ │  │  dashboard/      │
   │                  │  │              │  │                  │
   │  POST /predict   │  │  Evidently   │  │  Live predictor  │
   │  POST /predict/  │  │  drift_      │  │  Drift charts    │
   │    batch         │  │  monitor.py  │  │  Alert feed      │
   │  GET  /health    │  │              │  │  Prediction log  │
   │  GET  /model/    │  │  HTML reports│  │                  │
   │    info          │  │  MLflow log  │  │                  │
   └──────────────────┘  └──────────────┘  └──────────────────┘
```

---

## ⚠️ Why SMOTE on train only?

```python
# CORRECT ✓
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_res, y_res = smote.fit_resample(X_train, y_train)   # SMOTE here
model.fit(X_res, y_res)
model.evaluate(X_test, y_test)   # original distribution

# WRONG ✗  (data leakage!)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
```

Applying SMOTE before splitting leaks synthetic minority samples into the test set, inflating metrics. Always split first.

---

## 📊 Key Metrics Tracked (MLflow)

| Metric | Description |
|---|---|
| `roc_auc` | Area under ROC curve |
| `avg_precision` | Area under Precision-Recall curve (better for imbalanced data) |
| `f1_tuned` | F1 at tuned decision threshold |
| `optimal_threshold` | Threshold that maximises F1 |
| `tp / fp / fn / tn` | Full confusion matrix |
| `recall_tuned` | Fraud recall — minimising missed fraud |

---

## 🛠 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + model status |
| `/model/info` | GET | Features, threshold, artefact path |
| `/predict` | POST | Single transaction scoring |
| `/predict/batch` | POST | Batch scoring (max 1000) |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

---

## 🛠 Possible Extensions

- [ ] Kafka consumer for real-time streaming predictions
- [ ] LightGBM / CatBoost model comparison
- [ ] SHAP explainability endpoint (`/explain`)
- [ ] Rule-based blocklist layer on top of ML score
- [ ] Feedback loop: confirmed fraud → retrain trigger
- [ ] Docker + docker-compose for full stack
- [ ] GitHub Actions CI: pytest + model quality gate

---

## 📄 License

MIT © 2025

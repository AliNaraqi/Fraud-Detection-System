"""
dashboard/app.py
Fraud Detection Dashboard — monitoring, drift analysis, live predictions.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from monitoring.monitor_pipeline import load_monitoring_history, run_monitoring
from models.predictor import get_predictions_log

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSight",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #080C14;
    color: #CDD9F0;
}
.main .block-container { padding: 1.5rem 2rem; max-width: 1380px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; }

[data-testid="stSidebar"] { background: #0D1525 !important; border-right: 1px solid #1C2E4A; }
[data-testid="stSidebar"] * { color: #8BA3C7 !important; }
[data-testid="stSidebar"] h1 { color: #E8F1FF !important; font-size: 18px !important; }

.kpi { background: #0D1525; border: 1px solid #1C2E4A; border-radius: 10px;
       padding: 16px 20px; position: relative; overflow: hidden; }
.kpi::after { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.kpi-ok::after   { background:#00E5A0; }
.kpi-warn::after { background:#F5A623; }
.kpi-crit::after { background:#FF4757; }
.kpi-info::after { background:#4A90E2; }
.kpi-label { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:.12em;
             text-transform:uppercase; color:#4A6B9A; margin-bottom:6px; }
.kpi-val   { font-size:28px; font-weight:600; letter-spacing:-.02em; }
.kpi-ok   .kpi-val { color:#00E5A0; }
.kpi-warn .kpi-val { color:#F5A623; }
.kpi-crit .kpi-val { color:#FF4757; }
.kpi-info .kpi-val { color:#4A90E2; }
.kpi-sub  { font-family:'Space Mono',monospace; font-size:11px; color:#4A6B9A; margin-top:3px; }

.alert-card { background:#0D1525; border:1px solid #1C2E4A; border-left:3px solid;
              border-radius:6px; padding:10px 14px; margin:5px 0;
              font-family:'Space Mono',monospace; font-size:12px; }
.alert-CRITICAL { border-left-color:#FF4757; }
.alert-WARNING  { border-left-color:#F5A623; }
.alert-INFO     { border-left-color:#4A90E2; }
.alert-ts   { color:#4A6B9A; font-size:11px; }
.alert-type { font-weight:700; font-size:11px; }
.alert-CRITICAL .alert-type { color:#FF4757; }
.alert-WARNING  .alert-type  { color:#F5A623; }
.alert-INFO     .alert-type  { color:#4A90E2; }
.alert-msg { color:#8BA3C7; margin-top:3px; }

.section { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:.1em;
           text-transform:uppercase; color:#4A6B9A; border-bottom:1px solid #1C2E4A;
           padding-bottom:6px; margin:1.5rem 0 .8rem; }

.risk-low      { background:#0A2818; color:#00E5A0; border:1px solid #00E5A020;
                 padding:2px 10px; border-radius:20px; font-size:12px; }
.risk-medium   { background:#2B1F07; color:#F5A623; border:1px solid #F5A62320;
                 padding:2px 10px; border-radius:20px; font-size:12px; }
.risk-high     { background:#2B0C10; color:#FF4757; border:1px solid #FF475720;
                 padding:2px 10px; border-radius:20px; font-size:12px; }
.risk-critical { background:#3B0510; color:#FF1F30; border:1px solid #FF1F3040;
                 padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#080C14; }
::-webkit-scrollbar-thumb { background:#1C2E4A; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="#080C14", plot_bgcolor="#080C14",
    font=dict(family="Space Mono, monospace", color="#CDD9F0", size=11),
    margin=dict(l=40, r=10, t=30, b=30),
    xaxis=dict(gridcolor="#1C2E4A", linecolor="#1C2E4A", tickfont=dict(color="#4A6B9A")),
    yaxis=dict(gridcolor="#1C2E4A", linecolor="#1C2E4A", tickfont=dict(color="#4A6B9A")),
)

def kpi(label, value, sub="", state="info"):
    st.markdown(
        f'<div class="kpi kpi-{state}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡 FraudSight")
    st.markdown("*Fraud detection monitoring*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "🔍 Drift Analysis",
        "📡 Live Predictor",
        "📋 Prediction Log",
    ])
    st.markdown("---")
    if st.button("▶ Run Monitoring", use_container_width=True):
        with st.spinner("Running Evidently drift analysis…"):
            try:
                run_monitoring(use_drift_data=True)
                st.success("Monitoring complete!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def get_history(): return load_monitoring_history(50)

@st.cache_data(ttl=20)
def get_preds(): return get_predictions_log()

history = get_history()
preds_df = get_preds()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("## Fraud Detection — Overview")

    total_runs   = len(history)
    drift_runs   = sum(1 for r in history if r.get("drift_detected"))
    total_alerts = sum(len(r.get("alerts", [])) for r in history)
    crit_alerts  = sum(
        sum(1 for a in r.get("alerts", []) if a.get("level") == "CRITICAL")
        for r in history
    )
    last_ts = history[-1]["timestamp"][:16].replace("T", " ") if history else "—"
    latest_drift = history[-1].get("drift_share", 0) if history else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Monitor Runs",   str(total_runs),         "total",           "info")
    with c2: kpi("Drift Detected", str(drift_runs),         f"of {total_runs}","warn" if drift_runs else "ok")
    with c3: kpi("Total Alerts",   str(total_alerts),       f"{crit_alerts} critical", "crit" if crit_alerts else "warn" if total_alerts else "ok")
    with c4: kpi("Latest Drift",   f"{latest_drift:.0%}",   "feature drift share", "warn" if latest_drift>.3 else "ok")
    with c5: kpi("Last Run",       last_ts,                 "UTC",              "info")

    st.markdown("<br>", unsafe_allow_html=True)

    if history and len(history) > 1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section">Drift Share Over Time</div>', unsafe_allow_html=True)
            hdf = pd.DataFrame([
                {"run": i, "drift_share": r["drift_share"], "ts": r["timestamp"][:10]}
                for i, r in enumerate(history)
            ])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hdf["ts"], y=hdf["drift_share"],
                mode="lines+markers",
                line=dict(color="#F5A623", width=2),
                marker=dict(color="#F5A623", size=5),
                fill="tozeroy", fillcolor="rgba(245,166,35,0.07)",
            ))
            fig.add_hline(y=config.DRIFT_SHARE_THRESHOLD, line_dash="dot",
                          line_color="#FF4757", annotation_text="threshold",
                          annotation_font_color="#FF4757", annotation_font_size=9)
            fig.update_layout(**PT, height=220)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            st.markdown('<div class="section">Alerts Per Run</div>', unsafe_allow_html=True)
            adf = pd.DataFrame([
                {"ts": r["timestamp"][:10], "alerts": len(r.get("alerts", []))}
                for r in history
            ])
            fig2 = go.Figure(go.Bar(
                x=adf["ts"], y=adf["alerts"],
                marker_color=["#FF4757" if v > 0 else "#00E5A0" for v in adf["alerts"]],
            ))
            fig2.update_layout(**PT, height=220)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Recent alerts feed
    st.markdown('<div class="section">Recent Alerts</div>', unsafe_allow_html=True)
    all_alerts = [
        {**a, "ts": r["timestamp"][:16].replace("T"," ")}
        for r in history for a in r.get("alerts", [])
    ]
    if all_alerts:
        for a in reversed(all_alerts[-15:]):
            lvl = a.get("level","INFO")
            st.markdown(
                f'<div class="alert-card alert-{lvl}">'
                f'<span class="alert-ts">{a["ts"]}</span> &nbsp;'
                f'<span class="alert-type">{lvl}</span>'
                f'<div class="alert-msg">{a["message"]}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No alerts yet. Run the monitoring pipeline.")

    if not preds_df.empty:
        st.markdown('<div class="section">Prediction Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Histogram(
                x=preds_df["fraud_probability"],
                nbinsx=40,
                marker_color="#4A90E2",
                opacity=0.8,
            ))
            fig.update_layout(**PT, height=220, title="Fraud Probability Distribution",
                              xaxis_title="P(fraud)", yaxis_title="count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            if "risk_tier" in preds_df.columns:
                tier_counts = preds_df["risk_tier"].value_counts().reset_index()
                tier_colors = {"low":"#00E5A0","medium":"#F5A623","high":"#FF4757","critical":"#FF1F30"}
                fig2 = go.Figure(go.Bar(
                    x=tier_counts["risk_tier"],
                    y=tier_counts["count"],
                    marker_color=[tier_colors.get(t,"#4A90E2") for t in tier_counts["risk_tier"]],
                ))
                fig2.update_layout(**PT, height=220, title="Transactions by Risk Tier")
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Drift Analysis
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Drift Analysis":
    st.markdown("## Drift Analysis")

    if not history:
        st.info("No monitoring runs yet. Click **▶ Run Monitoring** in the sidebar.")
        st.stop()

    latest = history[-1]
    feat_scores = latest.get("feature_scores", {})

    col1, col2, col3 = st.columns(3)
    with col1: kpi("Drift Detected",   "YES" if latest["drift_detected"] else "NO",
                   f"run {latest['run_id']}", "crit" if latest["drift_detected"] else "ok")
    with col2: kpi("Drift Share",      f"{latest['drift_share']:.0%}",
                   "of features drifted", "warn" if latest["drift_share"] > 0 else "ok")
    with col3: kpi("Test Pass Rate",
                   f"{latest['tests_passed']}/{latest['tests_total']}",
                   "GE tests", "ok" if latest["tests_passed"] == latest["tests_total"] else "warn")

    if feat_scores:
        st.markdown('<div class="section">Per-Feature Drift Scores (Latest Run)</div>', unsafe_allow_html=True)
        fdf = pd.DataFrame([
            {"feature": k, "score": v, "drifted": k in latest.get("drifted_features", [])}
            for k, v in feat_scores.items()
        ]).sort_values("score", ascending=True)

        colors = ["#FF4757" if d else "#4A90E2" for d in fdf["drifted"]]
        fig = go.Figure(go.Bar(
            x=fdf["score"], y=fdf["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in fdf["score"]],
            textposition="outside",
        ))
        fig.add_vline(x=config.PSI_THRESHOLD, line_dash="dot",
                      line_color="#F5A623",
                      annotation_text="PSI threshold",
                      annotation_font_color="#F5A623",
                      annotation_font_size=9)
        fig.update_layout(**PT, height=360, xaxis_title="Drift Score")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # HTML report links
    reports = sorted(config.REPORTS_DIR.glob("data_drift_*.html"), reverse=True)
    if reports:
        st.markdown('<div class="section">Evidently HTML Reports</div>', unsafe_allow_html=True)
        for r in reports[:5]:
            st.markdown(f"📄 `{r.name}`")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Live Predictor
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📡 Live Predictor":
    st.markdown("## Live Transaction Scorer")
    st.markdown("*Enter transaction features to get an instant fraud risk score.*")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount          = st.number_input("Amount ($)", 0.01, 5000.0, 342.50)
            hour_of_day     = st.slider("Hour of day", 0, 23, 2)
            day_of_week     = st.slider("Day of week (0=Mon)", 0, 6, 6)
            merchant_cat    = st.selectbox("Merchant category", range(8),
                                           format_func=lambda x: ["grocery","gas","restaurant","online","travel","entertainment","atm","pharmacy"][x])
        with col2:
            country_code        = st.selectbox("Country", range(10),
                                               format_func=lambda x: ["US","CA","GB","DE","FR","MX","CN","BR","IN","AU"][x])
            dist_home           = st.number_input("Distance from home (km)", 0.0, 10000.0, 2400.0)
            txn_1h              = st.number_input("Txns last 1h", 0, 20, 4)
            txn_24h             = st.number_input("Txns last 24h", 0, 100, 18)
        with col3:
            avg_amt_7d  = st.number_input("Avg amount 7d ($)", 1.0, 2000.0, 85.0)
            is_intl     = st.selectbox("International?", [0, 1], index=1)
            is_new_dev  = st.selectbox("New device?", [0, 1], index=1)
            is_weekend  = st.selectbox("Weekend?", [0, 1], index=1)

        submitted = st.form_submit_button("🔍 Analyse Transaction", use_container_width=True)

    if submitted:
        velocity = round(txn_1h / max(txn_24h / 24.0, 0.01), 3)
        zscore   = round((amount - avg_amt_7d) / max(avg_amt_7d, 1.0), 4)

        payload = {
            "amount": amount, "hour_of_day": hour_of_day,
            "day_of_week": day_of_week, "merchant_category": merchant_cat,
            "country_code": country_code, "distance_from_home": dist_home,
            "transactions_last_1h": txn_1h, "transactions_last_24h": txn_24h,
            "avg_amount_7d": avg_amt_7d, "is_international": is_intl,
            "is_new_device": is_new_dev, "is_weekend": is_weekend,
            "velocity_ratio": velocity, "amount_zscore": zscore,
        }

        try:
            from models.predictor import predict_single
            result = predict_single(payload)

            prob    = result["fraud_probability"]
            tier    = result["risk_tier"]
            is_fraud = result["is_fraud"]

            tier_colors = {"low":"#00E5A0","medium":"#F5A623","high":"#FF4757","critical":"#FF1F30"}
            color = tier_colors.get(tier, "#4A90E2")

            st.markdown(f"""
            <div style="background:#0D1525;border:1px solid #1C2E4A;border-left:4px solid {color};
                        border-radius:8px;padding:20px 24px;margin-top:16px;">
              <div style="font-family:'Space Mono',monospace;font-size:10px;
                          letter-spacing:.1em;text-transform:uppercase;color:#4A6B9A;">
                Fraud Analysis Result
              </div>
              <div style="font-size:36px;font-weight:700;color:{color};margin:8px 0;">
                {prob:.2%}
              </div>
              <div style="font-size:14px;color:#8BA3C7;">
                Fraud probability &nbsp;·&nbsp;
                <span style="color:{color};font-weight:600;">{tier.upper()} RISK</span>
                &nbsp;·&nbsp;
                {"🚨 FLAGGED AS FRAUD" if is_fraud else "✅ CLASSIFIED LEGITIMATE"}
              </div>
              <div style="font-family:'Space Mono',monospace;font-size:11px;
                          color:#4A6B9A;margin-top:10px;">
                threshold={result['decision_threshold']:.3f} &nbsp;·&nbsp;
                velocity_ratio={velocity:.2f} &nbsp;·&nbsp;
                amount_zscore={zscore:.2f}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4A6B9A"},
                    "bar":  {"color": color},
                    "bgcolor": "#0D1525",
                    "steps": [
                        {"range": [0,  20], "color": "#0A2818"},
                        {"range": [20, 50], "color": "#1A1A08"},
                        {"range": [50, 80], "color": "#2B0C10"},
                        {"range": [80,100], "color": "#3B0510"},
                    ],
                    "threshold": {
                        "line": {"color": "#F5A623", "width": 2},
                        "thickness": 0.75,
                        "value": result["decision_threshold"] * 100,
                    },
                },
            ))
            fig.update_layout(paper_bgcolor="#080C14", font=dict(color="#CDD9F0"),
                              height=220, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except FileNotFoundError:
            st.error("Model not loaded. Run `python scripts/setup.py` first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Prediction Log
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Prediction Log":
    st.markdown("## Prediction Log")

    if preds_df.empty:
        st.info("No predictions logged yet. Use the Live Predictor or call the API.")
        st.stop()

    total = len(preds_df)
    fraud_count = int(preds_df["is_fraud_predicted"].sum()) if "is_fraud_predicted" in preds_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total Logged",  f"{total:,}",              "predictions", "info")
    with c2: kpi("Flagged Fraud", f"{fraud_count:,}",         "transactions","crit" if fraud_count > 0 else "ok")
    with c3: kpi("Fraud Rate",    f"{fraud_count/total:.2%}" if total else "0%", "in log", "warn" if fraud_count/total > 0.1 else "ok")
    with c4:
        avg_p = preds_df["fraud_probability"].mean() if "fraud_probability" in preds_df.columns else 0
        kpi("Avg Prob", f"{avg_p:.3f}", "mean fraud score", "info")

    st.markdown('<div class="section">Recent Predictions</div>', unsafe_allow_html=True)
    show_cols = [c for c in ["predicted_at","amount","is_fraud_predicted",
                             "fraud_probability","risk_tier"] if c in preds_df.columns]
    st.dataframe(
        preds_df[show_cols].tail(50).sort_values(
            "predicted_at" if "predicted_at" in preds_df.columns else show_cols[0],
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

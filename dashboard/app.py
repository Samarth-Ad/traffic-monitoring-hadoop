"""Bangalore Traffic Intelligence — interactive live dashboard.

A self-contained demo UI: pick the built-in dataset **or upload your own CSV**,
then watch the full analysis run live — data validation, the Hadoop MapReduce
aggregate, KNN volume prediction, and a streaming near-real-time anomaly feed.

Runs on the host with only streamlit + pandas + scikit-learn (no Spark, no
OSMnx, no Docker required):

    streamlit run dashboard/app.py

Accepted input: the headerless `lat,lon,day_of_week,hour,volume` format, or any
CSV whose columns map to the canonical schema (see common/schema.py).
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
# Host: packages live in the repo root (parent of dashboard/).
# Container: they are mounted beside app.py under /app.
ROOT = APP_DIR if (APP_DIR / "mapreduce").exists() else APP_DIR.parent
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.schema import load_any, load_any_text, validate  # noqa: E402
from ml.knn_predict import train_and_evaluate  # noqa: E402
from analytics.anomaly_detection import AnomalyDetector  # noqa: E402

st.set_page_config(page_title="Bangalore Traffic Intelligence", layout="wide", page_icon="🚦")

MAPPER = ROOT / "mapreduce" / "mapper.py"
REDUCER = ROOT / "mapreduce" / "reducer.py"


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def _validate_text(text: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = validate(load_any_text(text))
    return report.clean, report.rejected, report.reasons


def run_mapreduce(canonical: pd.DataFrame) -> pd.DataFrame:
    """Run the REAL Hadoop Streaming mapper|sort|reducer on this data."""
    text = canonical.to_csv(index=False)
    mapped = subprocess.run([sys.executable, str(MAPPER)], input=text,
                            capture_output=True, text=True, check=True)
    ordered = "\n".join(sorted(mapped.stdout.splitlines())) + "\n"
    reduced = subprocess.run([sys.executable, str(REDUCER)], input=ordered,
                             capture_output=True, text=True, check=True)
    return pd.read_csv(io.StringIO(reduced.stdout))


def _builtin_text() -> str:
    for cand in (config.SOURCE_DIR / "bangalore_traffic_raw.txt", config.SAMPLE_CSV, config.CANONICAL_CSV):
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    raise FileNotFoundError("No built-in dataset found; run python/prepare_dataset.py.")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚦 Traffic Intelligence")
st.sidebar.caption("Upload data → watch the Big Data pipeline run live.")

source = st.sidebar.radio("Data source", ["Built-in Bangalore dataset", "Upload my own CSV"])

raw_text: str | None = None
if source == "Upload my own CSV":
    up = st.sidebar.file_uploader("CSV / TXT", type=["csv", "txt"])
    if up is not None:
        raw_text = up.getvalue().decode("utf-8", errors="replace")
    st.sidebar.caption(
        "Expected: `lat,lon,day_of_week,hour,volume` (headerless), or a CSV with "
        "columns that map to them."
    )
else:
    try:
        raw_text = _builtin_text()
    except FileNotFoundError as exc:
        st.sidebar.error(str(exc))

st.sidebar.markdown("---")
k = st.sidebar.slider("KNN neighbours (k)", 1, 25, 7)
weights = st.sidebar.selectbox("KNN weighting", ["distance", "uniform"])
threshold = st.sidebar.slider("Anomaly threshold (robust z)", 2.0, 8.0, 3.5, 0.5)
batch_size = st.sidebar.slider("Stream batch size", 20, 500, 150, 10)
speed = st.sidebar.select_slider("Stream speed", ["fast", "medium", "slow"], value="medium")
inject = st.sidebar.checkbox("Inject a demo anomaly", value=True,
                             help="Adds one extreme batch so an alert is guaranteed")
SPEED_DELAY = {"fast": 0.05, "medium": 0.2, "slow": 0.5}[speed]

st.title("Bangalore Traffic Intelligence")
st.caption("Distributed analytics · MapReduce · KNN prediction · near-real-time anomaly detection")

if not raw_text:
    st.info("⬅️ Choose the built-in dataset or upload a CSV in the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------ load + validate
clean, rejected, reasons = _validate_text(raw_text)

if clean.empty:
    st.error("No valid rows after validation. Check the file format (see the sidebar hint).")
    if reasons:
        st.write("Rejection reasons:", reasons)
    st.stop()

tab_data, tab_mr, tab_knn, tab_anom = st.tabs(
    ["① Data & validation", "② MapReduce aggregate", "③ KNN prediction", "④ Live anomaly feed"]
)

# ------------------------------------------------------------------ ① data
with tab_data:
    c = st.columns(4)
    c[0].metric("Input rows", f"{len(clean) + len(rejected):,}")
    c[1].metric("Clean rows", f"{len(clean):,}")
    c[2].metric("Rejected", f"{len(rejected):,}")
    c[3].metric("Roads", f"{clean['road_id'].nunique():,}")
    if reasons:
        st.caption("Validation notes: " + ", ".join(f"{k2}={v}" for k2, v in reasons.items()))
    st.markdown("**Average volume by hour of day**")
    by_hour = clean.groupby("hour")["volume"].mean()
    st.line_chart(by_hour)
    st.markdown("**Sample of the canonical data**")
    st.dataframe(clean.head(200), use_container_width=True)

# ------------------------------------------------------------------ ② MapReduce
with tab_mr:
    st.caption("Produced by the real Hadoop Streaming job (`mapreduce/mapper.py` + `reducer.py`).")
    try:
        mr = run_mapreduce(clean)
    except subprocess.CalledProcessError as exc:
        st.error(f"MapReduce failed: {exc.stderr}")
        mr = None
    if mr is not None and not mr.empty:
        st.metric("Aggregated (road × hour) groups", f"{len(mr):,}")
        roads = sorted(mr["road_id"].unique())
        chosen = st.multiselect("Roads to plot", roads, default=roads[: min(4, len(roads))])
        shown = mr[mr["road_id"].isin(chosen)] if chosen else mr
        pivot = shown.pivot_table(index="time_bucket", columns="road_id", values="avg_volume")
        st.markdown("**Average volume by hour (per road)**")
        st.line_chart(pivot)
        st.dataframe(mr.sort_values(["road_id", "time_bucket"]), use_container_width=True)

# ------------------------------------------------------------------ ③ KNN
with tab_knn:
    if len(clean) < max(10, k + 1):
        st.warning(f"Need at least {max(10, k + 1)} rows for KNN; this dataset has {len(clean)}.")
    else:
        res = train_and_evaluate(clean, k=k, weights=weights, test_frac=0.2)
        knn, base = res["knn"], res["baseline"]
        c = st.columns(4)
        c[0].metric("KNN MAE", f"{knn.mae:.2f}")
        c[1].metric("KNN RMSE", f"{knn.rmse:.2f}")
        c[2].metric("KNN R²", f"{knn.r2:.3f}")
        improv = (base.mae - knn.mae) / base.mae * 100 if base.mae else 0.0
        c[3].metric("vs baseline MAE", f"{improv:+.1f}%", help=f"Baseline MAE {base.mae:.2f}")
        st.caption(
            f"Target: volume · features: road (one-hot) + cyclical hour/day · "
            f"replay-order holdout (train {len(clean)-knn.n_test}, test {knn.n_test})."
        )
        preds = res["predictions"].copy()
        st.markdown("**Actual vs predicted volume** (points on the diagonal are perfect)")
        st.scatter_chart(preds, x="actual_volume", y="predicted_volume", color="road_id", height=360)
        st.dataframe(preds, use_container_width=True)

# ------------------------------------------------------------------ ④ live anomaly
with tab_anom:
    st.caption(
        "Robust median/MAD detector. Baselines are learned from the dataset, then "
        "batches are replayed and scored live — alerts appear as they are found."
    )
    start = st.button("▶ Run live analysis", type="primary")
    if start:
        tmp = Path(tempfile.mkdtemp())
        det = AnomalyDetector(threshold=threshold, min_samples=5,
                              state_path=tmp / "state.json", alerts_path=tmp / "alerts.csv")
        det.fit_baseline(clean)

        batches = [clean.iloc[i:i + batch_size].copy() for i in range(0, len(clean), batch_size)]
        if inject:
            spike = clean.head(3).copy()
            spike["hour"], spike["day_of_week"], spike["volume"] = 2, 1, 99
            batches.append(spike)

        prog = st.progress(0.0, text="Starting stream…")
        mcols = st.columns(4)
        m_batch, m_rows, m_alerts, m_sev = (mcols[0].empty(), mcols[1].empty(),
                                            mcols[2].empty(), mcols[3].empty())
        chart_ph = st.empty()
        table_ph = st.empty()

        all_alerts: list[dict] = []
        per_batch: list[dict] = []
        total_rows = 0
        for i, b in enumerate(batches):
            label = "INJECTED-anomaly" if (inject and i == len(batches) - 1) else f"batch-{i:03d}"
            alerts = det.process_batch(b, label)
            all_alerts.extend(alerts)
            total_rows += len(b)
            per_batch.append({"batch": i, "alerts": len(alerts)})

            m_batch.metric("Batches", f"{i + 1}/{len(batches)}")
            m_rows.metric("Rows processed", f"{total_rows:,}")
            m_alerts.metric("Alerts", f"{len(all_alerts):,}")
            sev = sum(1 for a in all_alerts if a["severity"] == "severe")
            m_sev.metric("Severe", f"{sev:,}")
            chart_ph.bar_chart(pd.DataFrame(per_batch).set_index("batch"))
            if all_alerts:
                recent = pd.DataFrame(all_alerts)[
                    ["road_id", "day_of_week", "hour", "observed_volume",
                     "expected_volume", "deviation_score", "direction", "severity", "batch_id"]
                ]
                table_ph.dataframe(recent.tail(15), use_container_width=True)
            prog.progress((i + 1) / len(batches), text=f"Processed {label}")
            time.sleep(SPEED_DELAY)

        prog.progress(1.0, text="Done")
        if all_alerts:
            st.success(f"Stream complete — {len(all_alerts)} potential unusual-traffic events detected.")
            st.download_button("Download alerts CSV", pd.DataFrame(all_alerts).to_csv(index=False),
                               "alerts.csv", "text/csv")
        else:
            st.info("Stream complete — no anomalies above the threshold. Lower it or inject a demo anomaly.")
    else:
        st.info("Press **Run live analysis** to stream the data and watch anomalies surface.")

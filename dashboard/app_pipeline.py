"""Phase 6 — Streamlit traffic insights dashboard.

Reads Phase-3 analytics Parquet from /data/analytics/ and Phase-4/5 graph
outputs from /data/graph/. Everything is file-based so the dashboard needs
no live HDFS connection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

ANALYTICS_DIR = Path(os.environ.get("ANALYTICS_DIR", "/data/analytics"))
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", "/data/graph"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
ANOMALY_DIR = Path(os.environ.get("ANOMALY_DIR", str(ANALYTICS_DIR / "anomalies")))
MAPREDUCE_DIR = Path(os.environ.get("MAPREDUCE_DIR", str(ANALYTICS_DIR / "mapreduce")))

st.set_page_config(page_title="Bangalore Traffic Insights", layout="wide", page_icon="🚦")


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def section_header(title: str, note: str = "") -> None:
    st.markdown(f"### {title}")
    if note:
        st.caption(note)


def missing(label: str, path: Path) -> None:
    st.info(f"**{label}** not available yet — expected at `{path}`.")


# -------------------- Sidebar --------------------
st.sidebar.title("Bangalore Traffic")
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "1. HDFS (Phase 1)\n"
    "2. Streaming (Phase 2)\n"
    "3. Spark analytics (Phase 3)\n"
    "4. Road graph (Phase 4)\n"
    "5. Graph algorithms (Phase 5)\n"
    "6. **This dashboard**"
)
view = st.sidebar.radio(
    "View",
    [
        "Overview",
        "MapReduce aggregates",
        "KNN prediction",
        "Anomaly feed",
        "Volume",
        "Speed",
        "Congestion",
        "Peak windows",
        "Road network",
        "Shortest paths",
    ],
)

# -------------------- Overview --------------------
if view == "Overview":
    st.title("🚦 Bangalore Traffic — Big Data Insights")
    st.markdown(
        "Dockerized Hadoop 3.4.2 + Spark 3.5 + OSMnx road network. "
        "Analytics layer aggregates the Kaggle Bangalore traffic dataset by road and time; "
        "the graph layer runs PageRank, centrality and shortest-path on the OpenStreetMap "
        "drivable network for the city."
    )
    cols = st.columns(4)
    vol = load_parquet(ANALYTICS_DIR / "traffic_volume_by_road")
    spd = load_parquet(ANALYTICS_DIR / "avg_speed_by_road")
    cong = load_parquet(ANALYTICS_DIR / "congestion_by_road")
    pr = load_parquet(GRAPH_DIR / "pagerank.parquet")
    cols[0].metric("Roads tracked", f"{len(vol):,}" if vol is not None else "—")
    cols[1].metric("Mean speed km/h", f"{spd['avg_speed_kmh'].mean():.1f}" if spd is not None else "—")
    cols[2].metric("Avg congestion", f"{cong['avg_congestion'].mean():.1f}" if cong is not None else "—")
    cols[3].metric("Junctions in graph", f"{len(pr):,}" if pr is not None else "—")

    st.markdown("#### Prediction & anomaly detection")
    mcols = st.columns(4)
    knn_metrics = load_json(MODELS_DIR / "knn_metrics.json")
    alerts = load_csv(ANOMALY_DIR / "alerts.csv")
    mr = load_csv(MAPREDUCE_DIR / "volume_by_road_hour.csv")
    if knn_metrics:
        mcols[0].metric("KNN MAE", f"{knn_metrics['knn']['mae']:.2f}")
        mcols[1].metric("Baseline MAE", f"{knn_metrics['baseline']['mae']:.2f}")
        mcols[2].metric("KNN R²", f"{knn_metrics['knn']['r2']:.3f}")
    else:
        for i in range(3):
            mcols[i].metric(["KNN MAE", "Baseline MAE", "KNN R²"][i], "—")
    mcols[3].metric("Anomaly alerts", f"{len(alerts):,}" if alerts is not None else "—")
    if mr is not None:
        st.caption(f"MapReduce produced {len(mr):,} road×hour aggregate groups.")

# -------------------- Volume --------------------
elif view == "Volume":
    section_header("Traffic Volume by Road", "Phase 3.1 — Spark aggregation")
    df = load_parquet(ANALYTICS_DIR / "traffic_volume_by_road")
    if df is None:
        missing("Volume analytics", ANALYTICS_DIR / "traffic_volume_by_road")
    else:
        top_n = st.slider("Top N roads", 5, 50, 20)
        head = df.sort_values("total_volume", ascending=False).head(top_n)
        fig = px.bar(head, x="road", y="total_volume", color="area", title="Top roads by total volume")
        fig.update_layout(xaxis_tickangle=-40, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# -------------------- Speed --------------------
elif view == "Speed":
    section_header("Average Speed by Road", "Phase 3.2")
    df = load_parquet(ANALYTICS_DIR / "avg_speed_by_road")
    if df is None:
        missing("Speed analytics", ANALYTICS_DIR / "avg_speed_by_road")
    else:
        fig = px.histogram(df, x="avg_speed_kmh", nbins=30, title="Distribution of avg speed (km/h)")
        st.plotly_chart(fig, use_container_width=True)
        slow = df.sort_values("avg_speed_kmh").head(20)
        st.markdown("**Slowest 20 roads**")
        st.dataframe(slow, use_container_width=True)

# -------------------- Congestion --------------------
elif view == "Congestion":
    section_header("Congestion Hotspots", "Phase 3.3 — threshold-based flagging")
    df = load_parquet(ANALYTICS_DIR / "congestion_by_road")
    if df is None:
        missing("Congestion analytics", ANALYTICS_DIR / "congestion_by_road")
    else:
        fig = px.scatter(
            df, x="avg_congestion", y="congested_fraction",
            size="samples", color="area", hover_name="road",
            title="Congestion fraction vs. average",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.sort_values("avg_congestion", ascending=False), use_container_width=True)

# -------------------- Peak --------------------
elif view == "Peak windows":
    section_header("Peak Volume Windows per Road", "Phase 3.4 — top-3 time buckets")
    df = load_parquet(ANALYTICS_DIR / "peak_windows")
    if df is None:
        missing("Peak-window analytics", ANALYTICS_DIR / "peak_windows")
    else:
        fig = px.bar(df.head(60), x="road", y="avg_volume", color="peak_window",
                     title="Peak time windows", barmode="group")
        fig.update_layout(xaxis_tickangle=-40, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

# -------------------- Graph --------------------
elif view == "Road network":
    section_header("Road-Network Importance", "Phase 4 (graph) + Phase 5 PageRank")
    nodes = load_parquet(GRAPH_DIR / "nodes.parquet")
    pr = load_parquet(GRAPH_DIR / "pagerank.parquet")
    if nodes is None or pr is None:
        missing("Graph outputs", GRAPH_DIR)
    else:
        merged = nodes.merge(pr, on="node_id", how="inner")
        top = merged.sort_values("pagerank", ascending=False).head(300)
        center = [top["lat"].mean(), top["lon"].mean()]
        m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        max_pr = top["pagerank"].max()
        for _, row in top.iterrows():
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=3 + 10 * (row.pagerank / max_pr),
                color="crimson", fill=True, fill_opacity=0.6,
                popup=f"node {row.node_id}<br>pagerank={row.pagerank:.3e}",
            ).add_to(m)
        st_folium(m, height=600, use_container_width=True)
        st.markdown("**Top-20 junctions by PageRank**")
        st.dataframe(top.head(20)[["node_id", "lat", "lon", "pagerank"]], use_container_width=True)

# -------------------- Paths --------------------
elif view == "Shortest paths":
    section_header("Sampled Shortest Paths", "Phase 5 — Dijkstra on edge length")
    paths = load_parquet(GRAPH_DIR / "sample_shortest_path.parquet")
    if paths is None:
        missing("Shortest-path samples", GRAPH_DIR / "sample_shortest_path.parquet")
    else:
        st.dataframe(paths.sort_values("length_m"), use_container_width=True)
        fig = px.histogram(paths, x="length_m", nbins=20, title="Shortest-path length distribution (m)")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MapReduce --------------------
elif view == "MapReduce aggregates":
    section_header("Hadoop MapReduce — volume by road × hour", "Phase 2 — Hadoop Streaming job output")
    df = load_csv(MAPREDUCE_DIR / "volume_by_road_hour.csv")
    if df is None:
        missing("MapReduce output", MAPREDUCE_DIR / "volume_by_road_hour.csv")
    else:
        st.caption(f"{len(df):,} (road, hour) groups aggregated by the MapReduce job.")
        roads = sorted(df["road_id"].unique())
        chosen = st.multiselect("Roads", roads, default=roads[: min(4, len(roads))])
        shown = df[df["road_id"].isin(chosen)] if chosen else df
        fig = px.line(
            shown.sort_values("time_bucket"),
            x="time_bucket", y="avg_volume", color="road_id", markers=True,
            title="Average volume by hour of day (MapReduce aggregate)",
        )
        fig.update_layout(height=480, xaxis_title="hour", yaxis_title="avg volume")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.sort_values(["road_id", "time_bucket"]), use_container_width=True)

# -------------------- KNN prediction --------------------
elif view == "KNN prediction":
    section_header("KNN Traffic-Volume Prediction", "Phase 3 — scikit-learn KNN vs historical baseline")
    metrics = load_json(MODELS_DIR / "knn_metrics.json")
    preds = load_csv(MODELS_DIR / "knn_predictions.csv")
    if metrics is None or preds is None:
        missing("KNN model outputs", MODELS_DIR)
    else:
        c = st.columns(4)
        c[0].metric("KNN MAE", f"{metrics['knn']['mae']:.2f}", help="Mean absolute error on the replay-order holdout")
        c[1].metric("KNN RMSE", f"{metrics['knn']['rmse']:.2f}")
        c[2].metric("KNN R²", f"{metrics['knn']['r2']:.3f}")
        c[3].metric("MAE vs baseline", f"-{metrics.get('mae_improvement_pct_vs_baseline', 0):.1f}%",
                    help="Lower error than the per-road/hour historical mean")
        st.caption(
            f"Baseline MAE {metrics['baseline']['mae']:.2f} · "
            f"params k={metrics['params']['k']}, weights={metrics['params']['weights']}, "
            f"test_frac={metrics['params']['test_frac']}"
        )
        lim = max(preds["actual_volume"].max(), preds["predicted_volume"].max())
        fig = px.scatter(
            preds, x="actual_volume", y="predicted_volume", color="road_id",
            title="Actual vs predicted volume (test set)", opacity=0.7,
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="gray"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(preds, use_container_width=True)

# -------------------- Anomaly feed --------------------
elif view == "Anomaly feed":
    section_header("Near-Real-Time Anomaly Feed", "Phase 4 — robust median/MAD detector (beyond syllabus)")
    alerts = load_csv(ANOMALY_DIR / "alerts.csv")
    state = load_json(ANOMALY_DIR / "state.json")
    if alerts is None or alerts.empty:
        missing("Anomaly alerts", ANOMALY_DIR / "alerts.csv")
        if state:
            st.caption(
                f"Detector state present: {len(state.get('baselines', {}))} baselines, "
                f"{len(state.get('processed_batches', []))} batches processed, no alerts yet."
            )
    else:
        c = st.columns(4)
        c[0].metric("Total alerts", f"{len(alerts):,}")
        c[1].metric("Severe", int((alerts["severity"] == "severe").sum()))
        c[2].metric("High", int((alerts["severity"] == "high").sum()))
        if "latency_ms" in alerts:
            c[3].metric("Median latency ms", f"{alerts['latency_ms'].median():.1f}")
        sev_fig = px.bar(
            alerts.groupby(["severity", "direction"]).size().reset_index(name="count"),
            x="severity", y="count", color="direction", barmode="group",
            category_orders={"severity": ["moderate", "high", "severe"]},
            title="Alerts by severity and direction",
        )
        st.plotly_chart(sev_fig, use_container_width=True)
        st.dataframe(
            alerts.sort_values("deviation_score", key=lambda s: s.abs(), ascending=False),
            use_container_width=True,
        )

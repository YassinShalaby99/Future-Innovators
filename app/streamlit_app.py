
# streamlit_app.py
# Real-Time Traffic Analytics — Enterprise-Grade Single-File App

import os
import io
import json
import time
import math
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pyarrow as pa
import pyarrow.parquet as pq

st.set_page_config(
    page_title="Real-Time Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_COLUMNS = [
    "timestamp", "road_id", "lat", "lon", 
    "speed_kph", "volume", "occupancy", "direction", "segment_length_km"
]

def _np_rng(seed:int=42):
    rng = np.random.RandomState(seed)
    return rng

with st.sidebar:
    st.title("⚙️ Controls")
    st.write("Choose your live data source and refresh settings.")

    source = st.selectbox(
        "Data Source",
        options=["Simulated Stream", "Upload (CSV/Parquet)", "Cloud URL (CSV/Parquet)"],
        help="Pick where to read the incoming telemetry from."
    )

    refresh_sec = st.slider("Auto-refresh (seconds)", 0, 60, 10, help="0 disables auto-refresh.")
    window_minutes = st.slider("Rolling Window (minutes)", 5, 240, 60, help="Time window to keep in memory.")
    sim_city = st.selectbox("Simulation City", ["Cairo", "Alexandria", "Giza", "6th of October"], index=0)
    congestion_speed_threshold = st.number_input("Congestion Speed ≤ (kph)", min_value=1, max_value=120, value=35)
    anomaly_sensitivity = st.slider("Anomaly Sensitivity (IsolationForest)", 1, 100, 35, help="Higher = stricter (more anomalies).")
    enable_anomaly_detection = st.toggle("Enable Anomaly Detection", value=True)
    enable_eta_model = st.toggle("Enable ETA Model (RF)", value=True, help="Predict segment travel time using a simple RandomForest.")

    st.markdown("---")
    st.caption("Tip: Use **Cloud URL** with an Azure Blob SAS or GitHub raw link.")

# -------- Auto-refresh (safe) --------
if refresh_sec > 0:
    now = time.time()
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = now
    if now - st.session_state.last_refresh >= refresh_sec:
        st.session_state.last_refresh = now
        # bump query params to avoid cached fetches on some hosts
        st.experimental_set_query_params(_=int(now))
        st.experimental_rerun()

if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=DEFAULT_COLUMNS)

@st.cache_data(show_spinner=False)
def _read_cloud_url(url: str) -> pd.DataFrame:
    if url.endswith(".parquet") or "format=parquet" in url.lower():
        return pd.read_parquet(url, engine="pyarrow")
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def _read_bytes_to_df(b: bytes, file_name: str) -> pd.DataFrame:
    if file_name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(b), engine="pyarrow")
    return pd.read_csv(io.BytesIO(b))

CITY_CENTERS = {
    "Cairo": (30.0444, 31.2357),
    "Alexandria": (31.2001, 29.9187),
    "Giza": (30.0131, 31.2089),
    "6th of October": (29.9637, 30.9269),
}

def simulate_batch(n:int=1000, city:str="Cairo", seed:int=None) -> pd.DataFrame:
    rng = _np_rng(2025 if seed is None else seed)
    lat0, lon0 = CITY_CENTERS.get(city, CITY_CENTERS["Cairo"])
    road_ids = np.arange(100, 160)
    dirs = rng.choice(["N", "S", "E", "W"], size=road_ids.size)
    road_dirs = dict(zip(road_ids, dirs))
    road_pick = rng.choice(road_ids, size=n)
    base_lat = lat0 + rng.randn(n) * 0.05
    base_lon = lon0 + rng.randn(n) * 0.05
    rush = datetime.now().hour in [7,8,9,16,17,18]
    mean_speed = 28 if rush else 55
    speed_kph = np.clip(rng.normal(loc=mean_speed, scale=12, size=n), 3, 110)
    volume = np.clip(rng.normal(loc=60 if rush else 35, scale=15, size=n), 5, 160).astype(int)
    occupancy = np.clip((volume/160) + rng.normal(0.10, 0.05, size=n), 0.01, 0.98)
    seg_len = np.clip(rng.normal(1.2, 0.4, size=n), 0.2, 5.0)
    jam_mask = rng.rand(n) < (0.12 if rush else 0.06)
    speed_kph[jam_mask] = np.clip(speed_kph[jam_mask] - rng.rand(jam_mask.sum())*25, 3, None)
    occupancy[jam_mask] = np.clip(occupancy[jam_mask] + rng.rand(jam_mask.sum())*0.25, None, 0.99)
    timestamp = pd.to_datetime(datetime.now(timezone.utc)).round("s")
    df = pd.DataFrame({
        "timestamp": pd.date_range(timestamp, periods=n, freq="S"),
        "road_id": road_pick,
        "lat": base_lat,
        "lon": base_lon,
        "speed_kph": speed_kph.round(2),
        "volume": volume,
        "occupancy": occupancy.round(3),
        "direction": [road_dirs[r] for r in road_pick],
        "segment_length_km": seg_len.round(3)
    })
    return df

def acquire_data(source_mode: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        if source_mode == "Simulated Stream":
            df = simulate_batch(n=500, city=sim_city)
            return df, None
        elif source_mode == "Upload (CSV/Parquet)":
            up = st.sidebar.file_uploader("Upload CSV/Parquet", type=["csv","parquet"], key="upload_box")
            if up is not None:
                raw = up.read()
                df = _read_bytes_to_df(raw, up.name)
                return df, f"Uploaded: {up.name}"
            else:
                st.info("Upload a file to start, or switch to Simulated/Cloud source.")
                return pd.DataFrame(columns=DEFAULT_COLUMNS), None
        else:
            url = st.sidebar.text_input("Enter public URL (CSV/Parquet)", key="cloud_url", value="")
            if not url:
                st.warning("Provide a URL to fetch data from a cloud location (Azure/GitHub/etc.).")
                return pd.DataFrame(columns=DEFAULT_COLUMNS), None
            df = _read_cloud_url(url)
            return df, f"Cloud URL: {url[:80]}..."
    except Exception as e:
        st.exception(e)
        return pd.DataFrame(columns=DEFAULT_COLUMNS), f"Error: {e}"

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    colmap = {
        "time": "timestamp", "datetime": "timestamp",
        "lat": "lat", "latitude": "lat",
        "lon": "lon", "lng": "lon", "longitude": "lon",
        "speed": "speed_kph", "speed_km_h": "speed_kph",
        "count": "volume", "vol": "volume",
        "occ": "occupancy",
        "dir": "direction",
        "segment_km": "segment_length_km",
        "seg_len_km": "segment_length_km",
        "segment_length": "segment_length_km",
    }
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in colmap:
            ren[c] = colmap[lc]
    df = df.rename(columns=ren)
    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            if c == "timestamp":
                df[c] = pd.Timestamp.utcnow()
            elif c in ["lat","lon"]:
                df[c] = np.nan
            elif c == "direction":
                df[c] = "N"
            elif c == "segment_length_km":
                df[c] = 1.0
            else:
                df[c] = 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
    num_cols = ["speed_kph","volume","occupancy","segment_length_km","lat","lon"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["direction"] = df["direction"].astype(str)
    df["road_id"] = df["road_id"].astype(str)
    df = df.dropna(subset=["lat","lon"])
    return df[DEFAULT_COLUMNS].copy()

def append_to_buffer(new_df: pd.DataFrame, minutes: int):
    buf = st.session_state.buffer
    st.session_state.buffer = pd.concat([buf, new_df], ignore_index=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)
    st.session_state.buffer = st.session_state.buffer[st.session_state.buffer["timestamp"] >= cutoff]
    st.session_state.buffer.reset_index(drop=True, inplace=True)

def compute_metrics(df: pd.DataFrame, congestion_thr: float) -> dict:
    if df.empty:
        return {"avg_speed": 0.0,"median_speed": 0.0,"congestion_rate": 0.0,"total_volume": 0,"records": 0}
    avg_speed = float(df["speed_kph"].mean())
    med_speed = float(df["speed_kph"].median())
    congestion_rate = float((df["speed_kph"] <= congestion_thr).mean())
    total_volume = int(df["volume"].sum())
    return {"avg_speed": round(avg_speed,2),"median_speed": round(med_speed,2),"congestion_rate": round(congestion_rate,3),"total_volume": total_volume,"records": len(df)}

@st.cache_data(show_spinner=False)
def fit_anomaly_model(df: pd.DataFrame, sensitivity:int=35):
    if df.empty:
        return None
    feats = df[["speed_kph","volume","occupancy"]].copy()
    feats = feats.fillna(feats.median(numeric_only=True))
    contamination = max(0.01, min(0.25, sensitivity/200))
    model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    model.fit(feats.values)
    return model

def score_anomalies(df: pd.DataFrame, model) -> pd.DataFrame:
    if model is None or df.empty:
        out = df.copy()
        out["anomaly"] = False
        out["anomaly_score"] = 0.0
        return out
    feats = df[["speed_kph","volume","occupancy"]].copy()
    feats = feats.fillna(feats.median(numeric_only=True))
    scores = model.decision_function(feats.values)
    preds = model.predict(feats.values)
    out = df.copy()
    out["anomaly_score"] = scores
    out["anomaly"] = preds == -1
    return out

@st.cache_data(show_spinner=False)
def fit_eta_model(df: pd.DataFrame):
    if df.empty:
        return None
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["wday"] = work["timestamp"].dt.weekday
    work["travel_min"] = (work["segment_length_km"] / work["speed_kph"].replace(0, np.nan)) * 60
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["travel_min"])
    if work.shape[0] < 200:
        return None
    X = work[["volume","occupancy","hour","wday","segment_length_km"]]
    y = work["travel_min"]
    model = Pipeline([("scaler", StandardScaler()),("rf", RandomForestRegressor(n_estimators=200, random_state=42))])
    model.fit(X, y)
    return model

def predict_eta(df: pd.DataFrame, model) -> pd.DataFrame:
    if model is None or df.empty:
        out = df.copy()
        out["eta_min_pred"] = np.nan
        return out
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["wday"] = work["timestamp"].dt.weekday
    X = work[["volume","occupancy","hour","wday","segment_length_km"]]
    preds = model.predict(X)
    out = df.copy()
    out["eta_min_pred"] = np.round(preds, 2)
    return out

st.title("🚦 Real-Time Traffic Analytics")
st.caption("Enterprise-grade dashboard for live traffic telemetry — ingest, monitor, detect anomalies, and predict travel time.")

new_batch, source_desc = acquire_data(source)
new_batch = normalize_df(new_batch)
if not new_batch.empty:
    append_to_buffer(new_batch, window_minutes)
df = st.session_state.buffer.copy()

# Ensure columns exist when ML toggles are OFF (for tooltips/charts)
if "anomaly" not in df.columns:
    df["anomaly"] = False
    df["anomaly_score"] = 0.0
if "eta_min_pred" not in df.columns:
    df["eta_min_pred"] = np.nan

left, mid, right = st.columns([2,2,3])
with left:
    st.subheader("Status")
    st.write(f"**Source:** {source_desc or source}")
    st.write(f"**Window:** last {window_minutes} min")
    st.write(f"**Now (UTC):** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
with mid:
    m = compute_metrics(df, congestion_speed_threshold)
    st.metric("Avg Speed (kph)", m["avg_speed"])
    st.metric("Median Speed (kph)", m["median_speed"])
    st.metric("Congestion Rate", f"{m['congestion_rate']*100:.1f}%")
    st.metric("Total Volume", m["total_volume"])
    st.metric("Records", m["records"])
with right:
    st.info("Exports & Settings")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Clear Buffer"):
            st.session_state.buffer = pd.DataFrame(columns=DEFAULT_COLUMNS)
            st.success("Buffer cleared.")
            st.stop()
    with colB:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV", csv_bytes, file_name="traffic_snapshot.csv", mime="text/csv")
    with colC:
        sink = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, sink)
        st.download_button("⬇️ Parquet", sink.getvalue(), file_name="traffic_snapshot.parquet", mime="application/octet-stream")

st.markdown("---")

if enable_anomaly_detection:
    with st.spinner("Training anomaly model..."):
        anom_model = fit_anomaly_model(df, anomaly_sensitivity)
    df = score_anomalies(df, anom_model)

if enable_eta_model:
    with st.spinner("Training ETA model..."):
        eta_model = fit_eta_model(df)
    df = predict_eta(df, eta_model)

alerts = []
if not df.empty:
    if (df["speed_kph"] <= congestion_speed_threshold).mean() > 0.35:
        alerts.append("Widespread congestion across the network.")
    if enable_anomaly_detection and df["anomaly"].mean() > 0.10:
        alerts.append("Elevated anomaly rate detected (potential incidents).")

if alerts:
    for a in alerts:
        st.error(f"⚠️ {a}")
else:
    st.success("✅ Network operating within normal parameters.")

st.subheader("Network Map")
if df.empty:
    st.info("No data available yet.")
else:
    tooltip = {
        "html": "<b>Road:</b> {road_id}<br/><b>Speed:</b> {speed_kph} kph<br/><b>Vol:</b> {volume}<br/><b>Occ:</b> {occupancy}<br/><b>Anomaly:</b> {anomaly}<br/><b>ETA (min):</b> {eta_min_pred}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=50,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=20,
        get_fill_color=[
            "255 * (speed_kph <= %d or anomaly)" % congestion_speed_threshold,
            "140 * (speed_kph > %d and not anomaly)" % congestion_speed_threshold,
            "80"
        ],
    )
    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=11,
        pitch=0,
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)

st.subheader("KPIs & Trends")
if not df.empty:
    c1, c2 = st.columns(2)
    with c1:
        sp = df.sort_values("timestamp").groupby(pd.Grouper(key="timestamp", freq="1min"))["speed_kph"].mean().reset_index()
        fig1 = px.line(sp, x="timestamp", y="speed_kph", title="Average Speed per Minute")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        vr = df.sort_values("timestamp").groupby(pd.Grouper(key="timestamp", freq="1min"))["volume"].sum().reset_index()
        fig2 = px.bar(vr, x="timestamp", y="volume", title="Volume per Minute")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "eta_min_pred" in df.columns:
            by_road = df.groupby("road_id")["eta_min_pred"].mean().reset_index().sort_values("eta_min_pred", ascending=False).head(15)
            fig3 = px.bar(by_road, x="road_id", y="eta_min_pred", title="Top Segments by Predicted ETA")
            st.plotly_chart(fig3, use_container_width=True)
    with c4:
        if "anomaly" in df.columns:
            anom = df.groupby("road_id")["anomaly"].mean().reset_index().sort_values("anomaly", ascending=False).head(15)
            anom["anomaly_pct"] = (anom["anomaly"]*100).round(1)
            fig4 = px.bar(anom, x="road_id", y="anomaly_pct", title="Anomaly % by Segment")
            st.plotly_chart(fig4, use_container_width=True)

st.subheader("Data Explorer")
with st.expander("Peek current buffer"):
    st.dataframe(df.head(1000), use_container_width=True)

st.subheader("Diagnostics")
diag = {
    "python_version": f"{os.sys.version.split()[0]}",
    "rows_in_buffer": int(len(df)),
    "min_ts": str(df["timestamp"].min()) if not df.empty else None,
    "max_ts": str(df["timestamp"].max()) if not df.empty else None,
    "source": source_desc or source,
    "auto_refresh_sec": refresh_sec,
    "window_minutes": window_minutes,
    "anomaly_detection_enabled": enable_anomaly_detection,
    "eta_model_enabled": enable_eta_model,
}
st.json(diag)

st.caption("© 2025 Real-Time Traffic Analytics — built with ❤️ for Yassin")
import os
import io
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Traffic Simulation Dashboard", layout="wide")
st.title("🚦 Real-Time Traffic Simulation")

# --- Data source selection ---
# If DATA_URL is set via Streamlit Secrets (or env), we load from Azure.
# Otherwise we load from a local path (dev mode).
DATA_URL = os.getenv("DATA_URL", st.secrets.get("DATA_URL", ""))

# Sidebar controls
dev_path = st.sidebar.text_input("Processed path (dev only)", "data/traffic_with_time.csv")
speed_mult = st.sidebar.selectbox("Playback speed", [1, 2, 5, 10], index=1)
window_minutes = st.sidebar.slider("Visible window (minutes)", 5, 120, 30, 5)

# --- Data loading helpers ---
@st.cache_data(show_spinner="Loading dataset...")
def _read_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    content = io.BytesIO(r.content)
    if url.lower().endswith(".parquet"):
        df = pd.read_parquet(content)
    else:
        df = pd.read_csv(content, parse_dates=["timestamp"])
    return df

@st.cache_data(show_spinner="Loading dataset...")
def _read_local(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

def load_df(local_path: str, url: str) -> pd.DataFrame:
    if url:
        src = "Azure (SAS URL)"
        df = _read_from_url(url)
    else:
        src = f"Local file: {local_path}"
        df = _read_local(local_path)

    # Ensure sorted by time and clean index to avoid KeyError on .loc
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the dataset.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, src

# --- Load the data (Azure preferred, local fallback) ---
try:
    df, source_label = load_df(dev_path, DATA_URL)
    st.caption(f"Data source: **{source_label}**")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# --- Filters (only apply if columns exist) ---
def _safe_unique(series_name: str):
    return sorted(df[series_name].dropna().unique().tolist()) if series_name in df.columns else []

cities = st.sidebar.multiselect("Cities", _safe_unique("City"))
vehicles = st.sidebar.multiselect("Vehicle Type", _safe_unique("Vehicle Type"))
weathers = st.sidebar.multiselect("Weather", _safe_unique("Weather"))

mask = pd.Series(True, index=df.index)
if cities and "City" in df.columns:
    mask &= df["City"].isin(cities)
if vehicles and "Vehicle Type" in df.columns:
    mask &= df["Vehicle Type"].isin(vehicles)
if weathers and "Weather" in df.columns:
    mask &= df["Weather"].isin(weathers)

df = df[mask].reset_index(drop=True)

if df.empty:
    st.warning("No data after filters. Clear filters to continue.")
    st.stop()

# --- Playback state ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "paused" not in st.session_state:
    st.session_state.paused = False

# Clamp idx within bounds if filters changed
st.session_state.idx = int(max(0, min(st.session_state.idx, len(df) - 1)))

col1, col2, col3 = st.columns(3)
if col1.button("▶️ Play"):
    st.session_state.paused = False
if col2.button("⏸️ Pause"):
    st.session_state.paused = True
if col3.button("⏮️ Reset"):
    st.session_state.idx = 0

# --- Windowing ---
def get_window(i: int):
    # Use iloc (positional) to avoid KeyError when index labels are not 0..n-1
    current_ts = df.iloc[i]["timestamp"]
    window_start = current_ts - pd.Timedelta(minutes=window_minutes)
    view = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= current_ts)]
    return view, current_ts

view, now = get_window(st.session_state.idx)

# --- KPIs (compute only if columns exist) ---
k1, k2, k3 = st.columns(3)
avg_speed = f"{view['Speed'].mean():.1f}" if "Speed" in view.columns else "–"
k1.metric("Avg Speed", avg_speed)

cong = f"{view['Congestion Score'].mean():.1f}/100" if "Congestion Score" in view.columns else "–"
k2.metric("Congestion", cong)

events = int(view["Random Event Occurred"].sum()) if "Random Event Occurred" in view.columns else 0
k3.metric("Events", events)

# --- Charts (guard for missing columns) ---
st.subheader(f"Timeline up to: {now}")

if "Speed" in view.columns:
    st.line_chart(view.set_index("timestamp")["Speed"])
else:
    st.info("Speed column not found — skipping speed chart.")

if "Traffic Density" in view.columns:
    st.line_chart(view.set_index("timestamp")["Traffic Density"])
else:
    st.info("Traffic Density column not found — skipping density chart.")

st.dataframe(view.tail(50))

# --- Advance frame ---
if not st.session_state.paused:
    st.session_state.idx = min(st.session_state.idx + speed_mult, len(df) - 1)
    time.sleep(0.5)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

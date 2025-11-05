import os
import io
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from datetime import datetime
import pyarrow.parquet as pq
from typing import Tuple, Optional
from streamlit_autorefresh import st_autorefresh

# Optional import (only used if you provide account+key secrets)
try:
    from azure.storage.blob import BlobServiceClient
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Traffic Simulation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== STYLING =====
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚦 Real-Time Traffic Simulation</p>', unsafe_allow_html=True)


# ===== DATA LOADING HELPERS =====
@st.cache_data(show_spinner="Loading dataset from Azure Blob Storage...", ttl=600)
def _read_from_azure_blob(account: str, key: str, container: str, blob: str) -> pd.DataFrame:
    """
    Uses account+key to read a blob (CSV or Parquet) from Azure.
    This is the PREFERRED method for production deployments.
    """
    if not _AZURE_AVAILABLE:
        raise RuntimeError("azure-storage-blob not installed. Add it to requirements.txt")

    try:
        svc = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=key
        )
        client = svc.get_blob_client(container=container, blob=blob)
        
        # Download blob data
        data = client.download_blob().readall()
        buf = BytesIO(data)

        # Parse based on file extension
        if blob.lower().endswith(".parquet"):
            df = pq.read_table(buf).to_pandas()
        else:
            df = pd.read_csv(buf, parse_dates=["timestamp"])

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read from Azure Blob Storage: {str(e)}")


@st.cache_data(show_spinner="Loading dataset from SAS URL...", ttl=600)
def _read_from_url(url: str) -> pd.DataFrame:
    """
    Fallback method: Works with CSV or Parquet over HTTPS (SAS URL)
    """
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        content = io.BytesIO(r.content)
        
        if url.lower().endswith(".parquet"):
            df = pd.read_parquet(content)
        else:
            df = pd.read_csv(content, parse_dates=["timestamp"])
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read from SAS URL: {str(e)}")


@st.cache_data(show_spinner="Loading local dataset...", ttl=600)
def _read_local(path: str) -> pd.DataFrame:
    """
    Local development fallback
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures timestamp column exists and is properly formatted
    """
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the dataset.")
    
    # Ensure proper dtype
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    
    # Remove invalid timestamps and sort
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    return df


def load_df() -> Tuple[pd.DataFrame, str]:
    """
    Load data with priority:
    1) Azure Blob Storage (account + key) - PREFERRED
    2) SAS URL (fallback)
    3) Local file (development only)
    """
    # Priority 1: Azure Blob Storage with account key (PREFERRED)
    az_keys = ("AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_KEY", "AZURE_CONTAINER", "AZURE_BLOB")
    if all(k in st.secrets for k in az_keys):
        account = st.secrets["AZURE_STORAGE_ACCOUNT"]
        key = st.secrets["AZURE_STORAGE_KEY"]
        container = st.secrets["AZURE_CONTAINER"]
        blob = st.secrets["AZURE_BLOB"]
        
        df = _read_from_azure_blob(account, key, container, blob)
        return _normalize_timestamp(df), f"Azure Blob Storage ({account}/{container}/{blob})"
    
    # Priority 2: SAS URL (fallback)
    sas_url = os.getenv("DATA_URL", st.secrets.get("DATA_URL", ""))
    if sas_url:
        df = _read_from_url(sas_url)
        return _normalize_timestamp(df), "Azure (SAS URL)"
    
    # Priority 3: Local file (development)
    dev_path = st.secrets.get("LOCAL_DEV_PATH", "data/traffic_with_time.csv")
    if os.path.exists(dev_path):
        df = _read_local(dev_path)
        return _normalize_timestamp(df), f"Local file: {dev_path}"
    
    raise RuntimeError("No valid data source found. Please configure Azure credentials in Streamlit secrets.")


# ===== SIDEBAR CONTROLS =====
st.sidebar.header("⚙️ Control Panel")

# Playback settings
st.sidebar.subheader("Playback Settings")
speed_mult = st.sidebar.selectbox(
    "Playback Speed",
    options=[1, 2, 5, 10, 20],
    index=1,
    help="Multiplier for playback speed"
)

window_minutes = st.sidebar.slider(
    "Visible Window (minutes)",
    min_value=5,
    max_value=120,
    value=30,
    step=5,
    help="Time window to display in charts"
)

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh",
    value=True,
    help="Automatically advance through the timeline"
)


# ===== LOAD DATA =====
try:
    df, source_label = load_df()
    st.sidebar.success(f"✅ **Data Source:** {source_label}")
    st.sidebar.metric("Total Records", f"{len(df):,}")
    
    if len(df) > 0:
        time_range = df["timestamp"].max() - df["timestamp"].min()
        st.sidebar.metric("Time Span", f"{time_range.days}d {time_range.seconds//3600}h")
    
except Exception as e:
    st.error(f"❌ **Failed to load data:** {e}")
    st.info("Please ensure your Azure credentials are configured in Streamlit secrets.")
    st.stop()


# ===== FILTERS =====
st.sidebar.subheader("🔍 Filters")

def _safe_unique(series_name: str):
    """Safely get unique values from a column"""
    if series_name in df.columns:
        return sorted(df[series_name].dropna().unique().tolist())
    return []

# City filter
cities = st.sidebar.multiselect(
    "Cities",
    options=_safe_unique("City"),
    help="Filter by city"
)

# Vehicle type filter
vehicles = st.sidebar.multiselect(
    "Vehicle Type",
    options=_safe_unique("Vehicle Type"),
    help="Filter by vehicle type"
)

# Weather filter
weathers = st.sidebar.multiselect(
    "Weather Conditions",
    options=_safe_unique("Weather"),
    help="Filter by weather"
)

# Apply filters
mask = pd.Series(True, index=df.index)
if cities and "City" in df.columns:
    mask &= df["City"].isin(cities)
if vehicles and "Vehicle Type" in df.columns:
    mask &= df["Vehicle Type"].isin(vehicles)
if weathers and "Weather" in df.columns:
    mask &= df["Weather"].isin(weathers)

df_filtered = df[mask].reset_index(drop=True)

if df_filtered.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filters.")
    st.stop()

# Show filter results
if len(df_filtered) < len(df):
    st.sidebar.info(f"Showing {len(df_filtered):,} of {len(df):,} records")


# ===== PLAYBACK STATE =====
if "idx" not in st.session_state:
    st.session_state.idx = 0

if "paused" not in st.session_state:
    st.session_state.paused = not auto_refresh  # keep your original logic

# Safe boot — make sure app renders *once* before auto-play starts
if "booted" not in st.session_state:
    st.session_state.booted = True
    st.session_state.paused = True

# Clamp idx within bounds
st.session_state.idx = int(max(0, min(st.session_state.idx, len(df_filtered) - 1)))


# ===== PLAYBACK CONTROLS =====
st.subheader("🎮 Playback Controls")
col1, col2, col3, col4 = st.columns(4)

if col1.button("▶️ Play", use_container_width=True):
    st.session_state.paused = False

if col2.button("⏸️ Pause", use_container_width=True):
    st.session_state.paused = True

if col3.button("⏮️ Reset", use_container_width=True):
    st.session_state.idx = 0

if col4.button("⏭️ Skip Forward", use_container_width=True):
    st.session_state.idx = min(st.session_state.idx + 100, len(df_filtered) - 1)

# --- init once ---
if "last_update" not in st.session_state:
    st.session_state.last_update = 0.0

# Progress bar
progress = st.session_state.idx / max(len(df_filtered) - 1, 1)
st.progress(progress)
st.caption(f"Progress: {st.session_state.idx + 1:,} / {len(df_filtered):,} records ({progress*100:.1f}%)")


# ===== TIME WINDOW =====
def get_window(i: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Get data window for current index"""
    current_ts = df_filtered.iloc[i]["timestamp"]
    window_start = current_ts - pd.Timedelta(minutes=window_minutes)
    view = df_filtered[
        (df_filtered["timestamp"] >= window_start) & 
        (df_filtered["timestamp"] <= current_ts)
    ]
    return view, current_ts

view, current_time = get_window(st.session_state.idx)


# ===== KPI METRICS =====
st.subheader("📊 Key Performance Indicators")
st.caption(f"**Current Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Average Speed
if "Speed" in view.columns and len(view) > 0:
    avg_speed = view["Speed"].mean()
    kpi1.metric(
        "Average Speed",
        f"{avg_speed:.1f} km/h",
        delta=None
    )
else:
    kpi1.metric("Average Speed", "–")

# Congestion Score
if "Congestion Score" in view.columns and len(view) > 0:
    avg_congestion = view["Congestion Score"].mean()
    kpi2.metric(
        "Congestion Level",
        f"{avg_congestion:.1f}/100",
        delta=None
    )
else:
    kpi2.metric("Congestion Level", "–")

# Random Events
if "Random Event Occurred" in view.columns:
    event_count = int(view["Random Event Occurred"].sum())
    kpi3.metric("Events", event_count)
else:
    kpi3.metric("Events", "–")

# Traffic Density
if "Traffic Density" in view.columns and len(view) > 0:
    avg_density = view["Traffic Density"].mean()
    kpi4.metric(
        "Avg Density",
        f"{avg_density:.2f}",
        delta=None
    )
else:
    kpi4.metric("Avg Density", "–")


# ===== CHARTS =====
st.subheader("📈 Traffic Analytics")

# Speed over time
if "Speed" in view.columns and len(view) > 0:
    st.markdown("**Speed Over Time**")
    chart_data = view.set_index("timestamp")[["Speed"]]
    st.line_chart(chart_data, use_container_width=True)
else:
    st.info("Speed data not available")

# Traffic Density over time
if "Traffic Density" in view.columns and len(view) > 0:
    st.markdown("**Traffic Density Over Time**")
    chart_data = view.set_index("timestamp")[["Traffic Density"]]
    st.line_chart(chart_data, use_container_width=True)
else:
    st.info("Traffic Density data not available")

# Congestion Score over time
if "Congestion Score" in view.columns and len(view) > 0:
    st.markdown("**Congestion Score Over Time**")
    chart_data = view.set_index("timestamp")[["Congestion Score"]]
    st.area_chart(chart_data, use_container_width=True)


# ===== DATA TABLE =====
st.subheader("📋 Recent Data")
display_columns = [col for col in view.columns if col in [
    "timestamp", "City", "Vehicle Type", "Speed", 
    "Traffic Density", "Congestion Score", "Weather", "Random Event Occurred"
]]

if display_columns:
    st.dataframe(
        view[display_columns].tail(50),
        use_container_width=True,
        hide_index=True
    )
else:
    st.dataframe(view.tail(50), use_container_width=True)


# ===== AUTO-ADVANCE LOGIC (non-blocking) =====
has_rows = len(df) > 0

if not st.session_state.paused and has_rows and st.session_state.idx < len(df) - 1:
    # gentle refresh every 1000 ms to keep CPU low on Streamlit Cloud
    st_autorefresh(interval=1000, key="player-refresh")
    st.session_state.idx = min(st.session_state.idx + speed_mult, len(df) - 1)
elif has_rows and st.session_state.idx >= len(df) - 1:
    st.sidebar.success("✅ Reached end of timeline")
    st.session_state.paused = True




# ===== FOOTER =====
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
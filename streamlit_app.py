import time
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Traffic Simulation Dashboard", layout="wide")
st.title("🚦 Real-Time Traffic Simulation")

path = st.sidebar.text_input("Processed CSV path", "data/traffic_with_time.csv")
speed_mult = st.sidebar.selectbox("Playback speed", [1,2,5,10], index=1)
window_minutes = st.sidebar.slider("Visible window (minutes)", 5,120,30,5)

@st.cache_data
def load_df(p):
    # Ensure sorted by time and clean index to avoid KeyError on .loc
    df = pd.read_csv(p, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

try:
    df = load_df(path)
except Exception as e:
    st.error(f"❌ Failed to load file: {e}")
    st.stop()

# Filters
cities = st.sidebar.multiselect("Cities", sorted(df['City'].dropna().unique().tolist()))
vehicles = st.sidebar.multiselect("Vehicle Type", sorted(df['Vehicle Type'].dropna().unique().tolist()))
weathers = st.sidebar.multiselect("Weather", sorted(df['Weather'].dropna().unique().tolist()))

mask = pd.Series(True, index=df.index)
if cities:
    mask &= df['City'].isin(cities)
if vehicles:
    mask &= df['Vehicle Type'].isin(vehicles)
if weathers:
    mask &= df['Weather'].isin(weathers)

df = df[mask].reset_index(drop=True)

if df.empty:
    st.warning("No data after filters. Clear filters to continue.")
    st.stop()

# Playback state
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'paused' not in st.session_state:
    st.session_state.paused = False

# Clamp idx within bounds if filters changed
st.session_state.idx = int(max(0, min(st.session_state.idx, len(df)-1)))

col1, col2, col3 = st.columns(3)
if col1.button("▶️ Play"): 
    st.session_state.paused = False
if col2.button("⏸️ Pause"): 
    st.session_state.paused = True
if col3.button("⏮️ Reset"): 
    st.session_state.idx = 0

def get_window(i):
    # Use iloc (positional) to avoid KeyError when index labels are not 0..n-1
    current_ts = df.iloc[i]['timestamp']
    window_start = current_ts - pd.Timedelta(minutes=window_minutes)
    view = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_ts)]
    return view, current_ts

view, now = get_window(st.session_state.idx)

# KPIs
k1,k2,k3 = st.columns(3)
k1.metric("Avg Speed", f"{view['Speed'].mean():.1f}" if 'Speed' in view else "–")
k2.metric("Congestion", f"{view['Congestion Score'].mean():.1f}/100" if 'Congestion Score' in view else "–")
events = int(view['Random Event Occurred'].sum()) if 'Random Event Occurred' in view else 0
k3.metric("Events", events)

# Charts
st.subheader(f"Timeline up to: {now}")
st.line_chart(view.set_index('timestamp')['Speed'])
st.line_chart(view.set_index('timestamp')['Traffic Density'])
st.dataframe(view.tail(50))

# Advance frame
if not st.session_state.paused:
    st.session_state.idx = min(st.session_state.idx + speed_mult, len(df)-1)
    time.sleep(0.5)
    # Streamlit >=1.22: st.rerun(); older: experimental_rerun
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

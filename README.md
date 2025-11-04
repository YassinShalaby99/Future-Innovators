# Real-Time Traffic Analytics Dashboard

This project demonstrates real-time data processing using Python and Streamlit. It simulates continuous traffic events, processes data in-memory, and visualizes traffic conditions including vehicle flow, average speed, and congestion indicators.

The objective is to build intuition for streaming pipelines, rolling window analytics, and real-time dashboards without needing platforms like Kafka or Event Hubs. This architecture can later scale into production streaming systems.

## Features
- Live synthetic traffic event generator
- Rolling data window for near-real-time KPIs
- Speed trends, congestion metrics, per‑location breakdown
- Lightweight architecture suitable for learning and prototyping
- Deployable locally or via Streamlit Cloud

## Architecture
```
Data Generator → Preprocessing → Rolling Window Buffer → Streamlit Dashboard
```

| Layer | Description |
|---|---|
Event Source | Emits synthetic traffic events |
Preprocessing | Data cleaning and validation |
Window Buffer | Maintains recent N events for streaming analytics |
UI Layer | Live dashboard for metrics and charts |

## Tech Stack
- Python
- Streamlit
- Pandas

## Local Setup
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Deploy (Streamlit Cloud)
- Push repo to GitHub
- Create new Streamlit Cloud app
- Entry file: `app/streamlit_app.py`

## Next Steps
- Integrate Kafka / Azure Event Hub streams
- Add anomaly detection alerts
- Persist data snapshots (DuckDB / Parquet)
- Integrate geospatial maps

## Goal
This project builds foundational understanding for real‑time data systems and prepares for enterprise streaming pipelines.

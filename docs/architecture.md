# System Architecture

This project simulates a real-time traffic streaming pipeline and renders results through a Streamlit dashboard.

## Components
| Component | Purpose |
|---|---|
Event Generator | Produces synthetic traffic events |
Preprocessing | Cleans and validates incoming data |
Rolling Window | Maintains most recent events for live KPIs |
UI | Displays metrics, charts, and event stream |

## Flow Diagram
```
┌──────────────┐ → ┌────────────────┐ → ┌──────────────────┐ → ┌──────────────┐
| Event Source |   | Preprocessing  |   | Rolling Window   |   | Dashboard UI |
└──────────────┘   └────────────────┘   └──────────────────┘   └──────────────┘
```

## Key Concepts
- Real‑time event flow
- In‑memory rolling analytics
- Stateful streaming logic
- Low‑latency UI refresh

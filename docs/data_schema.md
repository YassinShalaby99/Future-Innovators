# Data Schema

| Field | Type | Description |
|---|---|---|
timestamp | datetime | Event timestamp |
city | string | City identifier |
road_id | string | Road segment identifier |
speed_kph | float | Vehicle speed |
lat / lon | float | Geolocation (optional) |

### Validation Rules
- Timestamps must be valid and increasing
- Speed values must be non‑negative
- Coordinates optional but validated when present

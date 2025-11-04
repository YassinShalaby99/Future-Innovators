# Traffic Simulation Project

## Run locally
```bash
pip install -r requirements.txt
python preprocess.py --input data/raw.csv --output data/traffic_with_time.csv
streamlit run app/streamlit_app.py
```

## Streamlit Cloud
Upload repo and set entry to:
```
app/streamlit_app.py
```

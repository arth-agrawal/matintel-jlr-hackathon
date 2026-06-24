# MatIntel — Material Intelligence Platform

Hackathon MVP:
Real material dataset → unified schema → yield strength model → confidence → recommendation → decision report.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If matminer download fails during hackathon, first run `python scripts/bootstrap_data.py` on a stable connection and keep the generated CSVs.

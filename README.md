# NBA Playoff Blueprint — Dashboard

**What this repo contains**
- `app.py` — Streamlit dashboard that predicts playoff probability from rank-based features.
- `train_model.py` — (Optional) script to build the `nba_playoff_forest.joblib` model from raw CSV.
- `visualizations/` — model plots used by the Insights page.
- `data/sample_modeling_data.csv` — small sample dataset for local testing.
- `models/` — location for the saved model file (not committed if large).

---

## Quickstart — run locally

1. Create and activate a virtual env (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

"""
config.py

Central place for paths so every module agrees on where things live. Nothing
in here should need changing when you add a new source -- new sources get
their own scraper/validator pair and a couple of lines in run_pipeline.py.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# What app.py actually reads (via data/data_loader.py and json.load in app.py).
# Publishing here is the ONLY thing that's allowed to change what Streamlit sees.
DATA_DIR = REPO_ROOT / "data"
BONDS_CONFIG_PATH = REPO_ROOT / "bonds_config.json"

CPI_CSV_PATH = DATA_DIR / "raw_cpi.csv"
NBP_CSV_PATH = DATA_DIR / "raw_nbp.csv"

# Raw, untouched, timestamped scrapes -- the audit trail. Never read by app.py.
SNAPSHOT_ROOT = REPO_ROOT / "snapshots"

# CSV column order/schema shared by raw_cpi.csv and raw_nbp.csv today.
PRICE_CSV_COLUMNS = ["Data", "Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie"]

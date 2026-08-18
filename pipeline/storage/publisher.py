"""
publisher.py

The single choke point that promotes validated data into the files app.py
actually reads. Nothing else in the pipeline should ever open data/*.csv or
bonds_config.json in write mode -- keeping that in one place is what makes
rollback a one-line operation later ("read yesterday's snapshot, call
publish_* again") instead of git archaeology.

Writes CSVs (not JSON) for CPI/NBP to match the schema data/data_loader.py
already expects (Data, Otwarcie, Najwyzszy, Najnizszy, Zamkniecie, comma-
separated, dates as YYYY-MM-DD). bonds_config.json stays JSON, matching how
app.py already loads it with json.load().
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import (
    CPI_CSV_PATH,
    NBP_CSV_PATH,
    BONDS_CONFIG_PATH,
    PRICE_CSV_COLUMNS,
)


def _backup(path: Path) -> Path | None:
    """Keeps one timestamped .bak alongside the file being overwritten, as a
    last-resort local safety net on top of git history (which already gives
    you full rollback -- this just makes the immediately-previous version
    trivially available without a git command)."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PRICE_CSV_COLUMNS)
    df = pd.read_csv(path, parse_dates=["Data"])
    return df


def read_existing_bonds(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def publish_price_csv(new_df: pd.DataFrame, path: Path) -> Path:
    """Merges new_df into whatever's already on disk (new rows win on date
    collisions -- e.g. a revised CPI print), sorts, writes atomically via a
    temp file + rename so a crash mid-write can't leave a half-written CSV
    behind for Streamlit to read."""
    existing = read_existing_csv(path)

    combined = pd.concat([existing, new_df[PRICE_CSV_COLUMNS]], ignore_index=True)
    combined["Data"] = pd.to_datetime(combined["Data"])
    combined = (
        combined.drop_duplicates(subset="Data", keep="last")
        .sort_values("Data")
        .reset_index(drop=True)
    )

    _backup(path)

    tmp_path = path.with_suffix(".tmp")
    combined.to_csv(tmp_path, index=False, date_format="%Y-%m-%d")
    tmp_path.replace(path)

    return path


def publish_bonds_config(new_config: dict, path: Path) -> Path:
    """Overwrites bonds_config.json with the scraped+validated terms, merged
    onto whatever's already there (so a partial scrape -- e.g. one bond's
    detail page failed to parse -- doesn't wipe out bonds that scraped fine
    last run). Strips internal '_source_symbol' provenance keys before
    writing, since app.py's BONDS_CONFIG only expects the fields it already
    reads (is_bonus, bonus_length, ... does_capitalise)."""
    existing = read_existing_bonds(path)
    merged = dict(existing)

    for name, bond in new_config.items():
        clean_bond = {k: v for k, v in bond.items() if not k.startswith("_")}
        merged[name] = clean_bond

    _backup(path)

    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)

    return path


def publish_cpi(new_df: pd.DataFrame) -> Path:
    return publish_price_csv(new_df, CPI_CSV_PATH)


def publish_nbp(new_df: pd.DataFrame) -> Path:
    return publish_price_csv(new_df, NBP_CSV_PATH)


def publish_bonds(new_config: dict) -> Path:
    return publish_bonds_config(new_config, BONDS_CONFIG_PATH)

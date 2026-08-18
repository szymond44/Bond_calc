"""
common.py (validators)

Shared sanity checks for the two CSV-style series (CPI, NBP reference rate).
Each check returns (ok: bool, message: str) so run_pipeline.py can log every
finding, not just the first failure.
"""

from __future__ import annotations

import pandas as pd


def check_no_date_gaps(df: pd.DataFrame, max_gap_days: int, label: str):
    dates = pd.to_datetime(df["Data"]).sort_values()
    gaps = dates.diff().dt.days.dropna()
    bad = gaps[gaps > max_gap_days]
    if not bad.empty:
        worst = bad.max()
        return False, f"{label}: found a {int(worst)}-day gap between publications (max allowed {max_gap_days})"
    return True, f"{label}: no date gaps beyond {max_gap_days} days"


def check_value_range(df: pd.DataFrame, column: str, lo: float, hi: float, label: str):
    out_of_range = df[(df[column] < lo) | (df[column] > hi)]
    if not out_of_range.empty:
        sample = out_of_range.iloc[0]
        return False, (
            f"{label}: {len(out_of_range)} row(s) outside plausible range [{lo}, {hi}] "
            f"-- e.g. {sample['Data']} = {sample[column]}"
        )
    return True, f"{label}: all values within [{lo}, {hi}]"


def check_jump_vs_last_known(new_df: pd.DataFrame, existing_df: pd.DataFrame, column: str, max_abs_jump: float, label: str):
    """Compares the newest new value against the newest existing value already
    in data/*.csv -- catches a scraper silently reading the wrong field
    (e.g. picking up a percentage-point column instead of a rate) even when
    the new value looks individually plausible."""
    if existing_df.empty or new_df.empty:
        return True, f"{label}: nothing to compare (empty series)"

    last_existing = existing_df.sort_values("Data").iloc[-1][column]
    last_new = new_df.sort_values("Data").iloc[-1][column]
    jump = abs(last_new - last_existing)

    if jump > max_abs_jump:
        return False, (
            f"{label}: latest value jumped {jump:.2f} vs. last published "
            f"({last_existing} -> {last_new}), exceeds max_abs_jump={max_abs_jump}"
        )
    return True, f"{label}: latest jump ({jump:.2f}) within tolerance"


def check_required_columns(df: pd.DataFrame, required: list[str], label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"{label}: missing required column(s): {missing}"
    return True, f"{label}: all required columns present"

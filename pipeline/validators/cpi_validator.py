"""
cpi_validator.py

Sanity gate for freshly-scraped CPI data before it's allowed to overwrite
data/raw_cpi.csv. Ranges are deliberately generous (wide enough to admit
2022-2023's real ~18% inflation spike) -- this is a plausibility net, not a
tight forecast band.
"""

from __future__ import annotations

import pandas as pd

from pipeline.validators.common import (
    check_no_date_gaps,
    check_value_range,
    check_jump_vs_last_known,
    check_required_columns,
)
from pipeline.config import PRICE_CSV_COLUMNS

# Poland's CPI (YoY %) has ranged roughly -1.6% (deflation, 2015) to ~18.4%
# (Feb 2023) in the available history. Padding both ends for headroom.
CPI_MIN_PCT = -5.0
CPI_MAX_PCT = 30.0

# CPI is published monthly with some lag; allow up to ~50 days between prints
# before flagging a gap (covers a slow month without being toothless).
CPI_MAX_GAP_DAYS = 50

# A single-month YoY CPI swing beyond 5 percentage points would be highly
# unusual outside a genuine crisis and is a reasonable "did the scraper read
# the wrong field" tripwire.
CPI_MAX_MONTHLY_JUMP = 5.0


def validate_cpi(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> tuple[bool, list[str]]:
    # Only range-check rows that are actually NEW relative to what's already
    # published. The scraper pulls GUS's full history back to 1982, which
    # genuinely includes real inflation over 30% (Poland's early-1980s
    # hyperinflation) -- that's true history, not a scraper bug, and
    # range-checking the whole re-scraped series would flag it every single
    # run, forever. What we actually want to know is: does the newest data
    # point (the thing this run would actually add) look sane. On the very
    # first run (existing_df empty), fall back to just the single latest row.
    if not existing_df.empty:
        cutoff = pd.to_datetime(existing_df["Data"]).max()
        rows_to_range_check = new_df[pd.to_datetime(new_df["Data"]) > cutoff]
    else:
        rows_to_range_check = new_df.sort_values("Data").tail(1)

    if rows_to_range_check.empty:
        range_check = (True, "CPI: no new rows since last publish, nothing to range-check")
    else:
        range_check = check_value_range(rows_to_range_check, "Zamkniecie", CPI_MIN_PCT, CPI_MAX_PCT, "CPI")

    checks = [
        check_required_columns(new_df, PRICE_CSV_COLUMNS, "CPI"),
        range_check,
        check_no_date_gaps(new_df, CPI_MAX_GAP_DAYS, "CPI"),
        check_jump_vs_last_known(new_df, existing_df, "Zamkniecie", CPI_MAX_MONTHLY_JUMP, "CPI"),
    ]
    messages = [msg for _, msg in checks]
    all_ok = all(ok for ok, _ in checks)
    return all_ok, messages

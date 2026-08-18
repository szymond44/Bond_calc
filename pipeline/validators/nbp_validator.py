"""
nbp_validator.py

Sanity gate for the NBP reference rate before it overwrites data/raw_nbp.csv.
"""

from __future__ import annotations

import pandas as pd

from pipeline.validators.common import (
    check_value_range,
    check_jump_vs_last_known,
    check_required_columns,
)
from pipeline.config import PRICE_CSV_COLUMNS

# NBP's reference rate has ranged from 24% (1998) down to 0.10% (2020-2021)
# in the available history.
NBP_MIN_PCT = 0.0
NBP_MAX_PCT = 30.0

# The rate only moves on MPC decisions -- typically 0.25-1.00pp per meeting.
# A single-step jump beyond 3pp is a plausibility tripwire, not a hard
# theoretical limit (emergency cuts/hikes have happened).
NBP_MAX_STEP_JUMP = 3.0

# NBP rate archive intentionally has no gap check: unlike CPI, this series
# only adds a row when the rate actually changes, so long flat periods (e.g.
# 2015-2020) are expected, not missing data.


def validate_nbp(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> tuple[bool, list[str]]:
    checks = [
        check_required_columns(new_df, PRICE_CSV_COLUMNS, "NBP"),
        check_value_range(new_df, "Zamkniecie", NBP_MIN_PCT, NBP_MAX_PCT, "NBP"),
        check_jump_vs_last_known(new_df, existing_df, "Zamkniecie", NBP_MAX_STEP_JUMP, "NBP"),
    ]
    messages = [msg for _, msg in checks]
    all_ok = all(ok for ok, _ in checks)
    return all_ok, messages

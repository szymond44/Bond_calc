"""
bonds_validator.py

Sanity gate for freshly-scraped bond terms before they're allowed to
overwrite bonds_config.json. Checks structure (required fields present, sane
types) and plausibility (rates/margins/penalties in a realistic range) --
does not check that the *specific* numbers match the market perfectly, since
that's exactly the thing this run might be legitimately updating.
"""

from __future__ import annotations

REQUIRED_FIELDS = {
    "is_bonus": int,
    "bonus_length": int,
    "bonus_rate": float,
    "margin": float,
    "index_type": str,
    "early_buyout_penalty": float,
    "timeframe_months": int,
    "capitalisation_period": int,
    "does_capitalise": bool,
}

VALID_INDEX_TYPES = {"fixed", "nbp", "cpi"}

# Plausibility bands -- generous, meant to catch parsing errors (e.g. reading
# "4,15%" as 4.15 instead of 0.0415), not to second-guess real market moves.
RATE_MIN, RATE_MAX = 0.0, 0.20        # 0% - 20% p.a.
MARGIN_MIN, MARGIN_MAX = 0.0, 0.05    # 0% - 5pp margin over index
PENALTY_MIN, PENALTY_MAX = 0.0, 0.05  # 0% - 5% early buyout penalty
TIMEFRAME_MIN, TIMEFRAME_MAX = 1, 180  # 1 month - 15 years


def validate_bonds(new_config: dict, existing_config: dict) -> tuple[bool, list[str]]:
    messages = []
    all_ok = True

    if not new_config:
        return False, ["Bonds: scraped config is empty -- refusing to publish an empty catalogue."]

    for name, bond in new_config.items():
        clean_bond = {k: v for k, v in bond.items() if not k.startswith("_")}

        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in clean_bond:
                all_ok = False
                messages.append(f"Bonds[{name}]: missing required field '{field}'")
                continue
            value = clean_bond[field]
            if expected_type is float and isinstance(value, int):
                value = float(value)  # int is an acceptable float in JSON-land
            if not isinstance(value, expected_type):
                all_ok = False
                messages.append(
                    f"Bonds[{name}]: field '{field}' expected {expected_type.__name__}, got {type(value).__name__}"
                )

        if clean_bond.get("index_type") not in VALID_INDEX_TYPES:
            all_ok = False
            messages.append(f"Bonds[{name}]: invalid index_type '{clean_bond.get('index_type')}'")

        for field, (lo, hi) in {
            "bonus_rate": (RATE_MIN, RATE_MAX),
            "margin": (MARGIN_MIN, MARGIN_MAX),
            "early_buyout_penalty": (PENALTY_MIN, PENALTY_MAX),
        }.items():
            value = clean_bond.get(field)
            if isinstance(value, (int, float)) and not (lo <= value <= hi):
                all_ok = False
                messages.append(f"Bonds[{name}]: {field}={value} outside plausible range [{lo}, {hi}]")

        timeframe = clean_bond.get("timeframe_months")
        if isinstance(timeframe, int) and not (TIMEFRAME_MIN <= timeframe <= TIMEFRAME_MAX):
            all_ok = False
            messages.append(f"Bonds[{name}]: timeframe_months={timeframe} outside plausible range")

        # If this bond existed before, flag (don't block) a large swing in its
        # headline first-period rate  worth a human glance, not necessarily wrong.
        if name in existing_config:
            old_rate = existing_config[name].get("bonus_rate")
            new_rate = clean_bond.get("bonus_rate")
            if isinstance(old_rate, (int, float)) and isinstance(new_rate, (int, float)):
                if abs(new_rate - old_rate) > 0.02:
                    messages.append(
                        f"Bonds[{name}]: NOTE bonus_rate moved {old_rate} -> {new_rate} "
                        f"(>2pp change, not blocking -- verify it's a real rate change)"
                    )

    if all_ok:
        messages.append(f"Bonds: {len(new_config)} bond(s) passed structural + plausibility checks")

    return all_ok, messages

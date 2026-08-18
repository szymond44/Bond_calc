"""
gov_pl_bonds_scraper.py

Scrapes current retail Treasury bond terms from the Ministry of Finance's
own "Biezaca oferta" (current offer) page -- the official primary source,
rather than obligacjeskarbowe.pl (PKO BP's retail distribution portal).

Source: https://www.gov.pl/web/finanse/biezaca-oferta2

Why this is the better source:
  - It's the Ministry's own page, not a distributor's -- lowest legal/ToS
    risk of the options considered.
  - The page footer explicitly licenses its text content under
    Creative Commons CC BY-SA 4.0 ("Tresci tekstowe publikowane w serwisie
    (...) sa udostepniane na licencji (...) CC BY-SA 4.0"), so reuse here
    is on solid ground.
  - It's one page covering all 8 bond types in plain prose, updated monthly
    -- no per-symbol detail pages to crawl, fewer places for a scraper to
    silently break.

>>> ONE GAP: early buyout penalty <<<
The current-offer page does not publish the early buyout fee
(oplata za przedterminowy wykup) -- that lives in separate "warunki emisji"
(terms of issuance) PDF letters linked at the bottom of the same page, one
per bond series, which are not structured text and change rarely. Rather
than fragile-parse a PDF every run for a number that essentially never
moves, EARLY_BUYOUT_PENALTY_DEFAULTS below holds the current known values as
a maintained constant. Re-verify these by hand against
https://www.gov.pl/web/finanse/obligacje-detaliczne1 (or the linked "listy
emisyjne" PDFs) every few months, or whenever a bond family's terms change --
they're structural properties of each bond family, not something that
should silently drift without a human noticing.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup

from pipeline.scrapers.base_scraper import BaseScraper, ScraperError

OFFER_URL = "https://www.gov.pl/web/finanse/biezaca-oferta2"

POLISH_MONTHS = {
    "styczeń": 1, "luty": 2, "marzec": 3, "kwiecień": 4, "maj": 5, "czerwiec": 6,
    "lipiec": 7, "sierpień": 8, "wrzesień": 9, "październik": 10, "listopad": 11, "grudzień": 12,
}

TIMEFRAME_UNIT_TO_MONTHS = {
    "miesięczne": 1,
    "roczne": 12,
    "letnie": 12,
    "letni": 12,
    "letnia": 12,
}

# Structural constants not published on the current-offer page (see module
# docstring). Family -> decimal fraction of face value charged on early
# redemption.
#
# ROD verified 2026-08 against the real "list emisyjny" (terms of issuance)
# for ROD0738 (list nr 68/2026): point 19.5 caps the deduction at 3.00 PLN
# per 100 PLN face-value bond -> 3.00/100.00 = 0.03. The original
# bonds_config.json this pipeline was built against had 0.02 for ROD -- that
# was never actually checked against a primary source before now, it was
# just carried forward as presumed-correct. Corrected here; the original
# bonds_config.json value should be corrected too. The other 7 families below
# are still UNVERIFIED against a primary document -- treat them the same way
# ROD just was, not as already-confirmed.
EARLY_BUYOUT_PENALTY_DEFAULTS = {
    "OTS": 0.0,
    "ROR": 0.005,
    "DOR": 0.007,
    "TOS": 0.01,
    "COI": 0.02,
    "EDO": 0.03,
    "ROS": 0.007,
    "ROD": 0.03,   
}


def _to_decimal_pct(text: str) -> Optional[float]:
    match = re.search(r"(\d{1,2},\d{1,2})\s?%", text)
    if not match:
        return None
    return round(float(match.group(1).replace(",", ".")) / 100.0, 6)


def _parse_timeframe_months(header_text: str) -> Optional[int]:
    """'3-miesięczne' -> 3, '1-roczne' -> 12, '2-letnie' -> 24, '10-letnie' -> 120."""
    match = re.search(r"(\d{1,2})[\s-]*(miesięczne|roczne|letnie|letni[aeą]?)", header_text)
    if not match:
        return None
    count, unit = match.groups()
    per_unit_months = TIMEFRAME_UNIT_TO_MONTHS.get(unit)
    if per_unit_months is None:
        return None
    return int(count) * per_unit_months


class GovPlBondsScraper(BaseScraper):
    source_name = "bonds_gov_pl"

    def _select_current_month_section(self, full_text: str) -> str:
        """The page lists 2+ consecutive months' offers (e.g. lipiec + sierpień).
        Splits on 'Oferta na <miesiąc> <rok> r.' headers and picks the most
        recent section whose date isn't in the future -- i.e. the currently
        active offer, not next month's if it's already been published early."""
        header_re = re.compile(r"Oferta na\s+(\w+)\s+(\d{4})\s*r\.")
        matches = list(header_re.finditer(full_text))
        if not matches:
            raise ScraperError(
                "Could not find any 'Oferta na <miesiąc> <rok> r.' section headers -- "
                "gov.pl page structure may have changed."
            )

        today = date.today()
        candidates = []
        for i, m in enumerate(matches):
            month_name, year_str = m.group(1).lower(), m.group(2)
            month_num = POLISH_MONTHS.get(month_name)
            if month_num is None:
                continue
            section_date = date(int(year_str), month_num, 1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            candidates.append((section_date, full_text[start:end]))

        if not candidates:
            raise ScraperError("Found offer headers but none had a parseable Polish month name.")

        not_future = [c for c in candidates if c[0] <= date(today.year, today.month, 1)]
        chosen_date, chosen_text = max(not_future or candidates, key=lambda c: c[0])
        self.logger.info("Using offer section for %s-%02d", chosen_date.year, chosen_date.month)
        return chosen_text

    def _parse_bond_blocks(self, section_text: str) -> dict:
        # Anchor on the HEADER PHRASE + SYMBOL together, not the symbol alone.
        # The live page turned out to also contain an unrelated "exchange your
        # maturing bonds" notice (e.g. "...obligacji wykupywanych w sierpniu -
        # serie OTS0826, ROR0826, DOR0826...") which mentions old, expiring
        # bond symbols in plain prose with no "X-miesieczne/roczne/letnie"
        # phrase nearby. A bare "3 letters + 4 digits" regex can't tell that
        # apart from a real offer heading and was matching both -- caught by
        # testing against real page text (again). Requiring the timeframe
        # phrase immediately before the symbol rules those false matches out,
        # since that phrase only appears in genuine per-bond offer headers.
        header_with_symbol_re = re.compile(
            r"\d{1,2}[\s-]*(?:miesięczne|roczne|letnie|letni[aeą]?)"
            r"[\s\S]{0,200}?\b([A-Z]{3}\d{4})\b"
        )
        header_matches = list(header_with_symbol_re.finditer(section_text))
        if not header_matches:
            raise ScraperError("No 'X-miesieczne/roczne/letnie ... SYMBOL' headers found in the selected offer section.")

        parsed = {}
        for i, hm in enumerate(header_matches):
            symbol = hm.group(1)
            family = symbol[:3]

            block_start = hm.start()
            block_end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(section_text)
            block = section_text[block_start:block_end]
            block = section_text[block_start:block_end]

            timeframe_months = _parse_timeframe_months(block)
            if timeframe_months is None:
                self.logger.warning("Could not parse timeframe for %s, skipping", symbol)
                continue

            rate_type_match = re.search(r"Oprocentowanie obligacji jest\s+(stałe|zmienne)", block)
            rate_type = rate_type_match.group(1) if rate_type_match else None

            if rate_type == "stałe" and "inflacji" not in block:
                index_type = "fixed"
            elif "stopy referencyjnej NBP" in block:
                index_type = "nbp"
            elif "inflacji" in block or "towarów i usług konsumpcyjnych" in block:
                index_type = "cpi"
            else:
                self.logger.warning("Could not determine index_type for %s, skipping", symbol)
                continue

            first_rate = _to_decimal_pct(block)
            if first_rate is None:
                self.logger.warning("Could not parse headline rate for %s, skipping", symbol)
                continue

            margin_match = re.search(r"marż[aęy]\s+w wysokości\s+(\d{1,2},\d{1,2})\s?%", block)
            margin = round(float(margin_match.group(1).replace(",", ".")) / 100.0, 6) if margin_match else 0.0

            does_capitalise = "kapitalizowane co" in block

            if "wypłacane co miesiąc" in block:
                capitalisation_period = 1
            elif "wypłacane co roku" in block or "kapitalizowane co roku" in block:
                capitalisation_period = 12
            elif timeframe_months <= 3:
                capitalisation_period = timeframe_months  # OTS: single period covering the whole life
            else:
                self.logger.warning("Could not determine capitalisation period for %s, skipping", symbol)
                continue

            # First-period ("bonus") length: for variable-rate bonds it's one
            # capitalisation period; for fixed-rate bonds the headline rate
            # holds for the bond's whole life.
            bonus_length = timeframe_months if rate_type == "stałe" else capitalisation_period

            early_buyout_penalty = EARLY_BUYOUT_PENALTY_DEFAULTS.get(family)
            if early_buyout_penalty is None:
                self.logger.warning("No known early_buyout_penalty default for family '%s' (symbol %s)", family, symbol)
                early_buyout_penalty = 0.0

            years = timeframe_months // 12
            config_key = f"{family}_{years}Y" if years >= 1 else f"{family}_{timeframe_months}M"

            parsed[config_key] = {
                "is_bonus": 1,
                "bonus_length": bonus_length,
                "bonus_rate": first_rate,
                "margin": margin,
                "index_type": index_type,
                "early_buyout_penalty": early_buyout_penalty,
                "timeframe_months": timeframe_months,
                "capitalisation_period": capitalisation_period,
                "does_capitalise": does_capitalise,
                "_source_symbol": symbol,
            }

        return parsed

    def run(self) -> dict:
        html = self.fetch_text(OFFER_URL)
        self.save_snapshot(html)

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text("\n", strip=True).replace("*", "")

        section_text = self._select_current_month_section(full_text)
        config_updates = self._parse_bond_blocks(section_text)

        if not config_updates:
            raise ScraperError("Parsed zero bonds successfully from gov.pl -- nothing safe to publish this run.")

        self.logger.info("Parsed terms for %d bonds: %s", len(config_updates), list(config_updates))
        return config_updates


if __name__ == "__main__":
    import json
    scraper = GovPlBondsScraper()
    result = scraper.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))

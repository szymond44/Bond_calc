"""
nbp_rate_scraper.py

Pulls the NBP reference rate ("stopa referencyjna") from NBP's own static
XML archive and normalizes it to the same shape as data/raw_nbp.csv
(Data, Otwarcie, Najwyzszy, Najnizszy, Zamkniecie).

Source: https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml
  - Official NBP data, generally reusable, no auth required.
  - This is a full historical archive (every rate change since 1998), not
    just the latest value -- one call gets you the whole series, no need to
    poll daily. It's tied to Monetary Policy Council decisions, not a fixed
    schedule, so re-fetching monthly is more than enough.
  - <pozycje obowiazuje_od="YYYY-MM-DD"> = the date a new rate set took
    effect. <pozycja id="ref" oprocentowanie="X,XX" /> is the reference rate
    (the one feeding your "nbp"-indexed bonds). Other ids seen in the archive:
    lom (lombard), dep (deposit), red (rediscount), dys (discount) -- not
    used by this model today but parsed here too in case you need them later.

Note this only has one rate per change-date (a step function), so
Otwarcie/Najwyzszy/Najnizszy/Zamkniecie are all set equal to that day's rate,
matching how raw_nbp.csv already stores it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd

from pipeline.scrapers.base_scraper import BaseScraper, ScraperError

NBP_ARCHIVE_URL = "https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml"

# Which <pozycja id="..."> to extract as the primary series this scraper returns.
# "ref" = stopa referencyjna, the one used by ROR_1Y / DOR_2Y in bonds_config.json.
PRIMARY_RATE_ID = "ref"

ALL_RATE_IDS = {
    "ref": "reference",
    "lom": "lombard",
    "dep": "deposit",
    "red": "rediscount",
    "dys": "discount",
}


class NbpRateScraper(BaseScraper):
    source_name = "nbp_rate"

    def run(self) -> pd.DataFrame:
        xml_bytes = self.fetch_bytes(NBP_ARCHIVE_URL, polite_delay=0.5)
        self.save_snapshot(xml_bytes.decode("utf-8-sig"))

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            raise ScraperError(f"Could not parse NBP rate archive XML: {e}") from e

        rows = []
        for pozycje in root.findall("pozycje"):
            effective_date = pozycje.get("obowiazuje_od")
            if not effective_date:
                continue

            rate_by_id = {}
            for pozycja in pozycje.findall("pozycja"):
                rid = pozycja.get("id")
                val = pozycja.get("oprocentowanie")
                if rid and val:
                    # NBP uses a comma decimal separator, e.g. "4,25"
                    rate_by_id[rid] = float(val.replace(",", "."))

            if PRIMARY_RATE_ID not in rate_by_id:
                self.logger.warning("No '%s' rate on %s, skipping", PRIMARY_RATE_ID, effective_date)
                continue

            value = rate_by_id[PRIMARY_RATE_ID]
            rows.append({
                "Data": pd.Timestamp(effective_date),
                "Otwarcie": value,
                "Najwyzszy": value,
                "Najnizszy": value,
                "Zamkniecie": value,
            })

        if not rows:
            raise ScraperError("Parsed zero rate rows from NBP archive -- source format may have changed.")

        df = pd.DataFrame(rows).drop_duplicates(subset="Data").sort_values("Data").reset_index(drop=True)
        self.logger.info(
            "Parsed %d reference-rate rows (%s .. %s)",
            len(df), df["Data"].min().date(), df["Data"].max().date(),
        )
        return df

#test
if __name__ == "__main__":
    scraper = NbpRateScraper()
    df = scraper.run()
    print(df.tail())
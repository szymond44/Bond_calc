from __future__ import annotations

import csv
import io
import ssl
import urllib.request
from typing import Optional

import pandas as pd

from pipeline.scrapers.base_scraper import BaseScraper, ScraperError

GUS_CSV_URL = "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultstronaopisowa/4741/1/1/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982_roku__2_2.csv"


class GusCpiScraper(BaseScraper):
    source_name = "gus_cpi"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> pd.DataFrame:
        
        # Use standard urllib with a User-Agent to prevent basic government firewall blocks
        req = urllib.request.Request(GUS_CSV_URL, headers={'User-Agent': 'Mozilla/5.0'})
        
        ssl_context = ssl._create_unverified_context()
        
        try:
            with urllib.request.urlopen(req, context=ssl_context) as response:
                raw_bytes = response.read()
        except Exception as e:
            raise ScraperError(f"Failed to download GUS CSV: {e}")

        try:
            text = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw_bytes.decode('windows-1250')


        self.save_snapshot(text)

        reader = csv.DictReader(io.StringIO(text), delimiter=';')
        rows = []

        for row in reader:
            if (row.get('Nazwa zmiennej') == 'Wskaźnik cen towarów i usług konsumpcyjnych' and
                row.get('Jednostka terytorialna') == 'Polska' and
                row.get('Sposób prezentacji') == 'Analogiczny miesiąc poprzedniego roku = 100'):

                year = int(row['Rok'])
                month = int(row['Miesiąc'])

                date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                
                # GUS YoY CPI is an index (e.g., 102.4 for +2.4%).
                # Convert to the "2.4 means 2.4%" convention used in raw_cpi.csv.
                try:
                    raw_val = float(row['Wartość'].replace(',', '.'))
                except ValueError:
                    continue  # Skip rows with missing or malformed values

                pct_value = round(raw_val - 100.0, 2)

                rows.append({
                    "Data": date,
                    "Otwarcie": pct_value,
                    "Najwyzszy": pct_value,
                    "Najnizszy": pct_value,
                    "Zamkniecie": pct_value,
                })

        if not rows:
            raise ScraperError("Parsed zero valid CPI rows from GUS CSV response -- check file structure or URL.")

        df = pd.DataFrame(rows).drop_duplicates(subset="Data").sort_values("Data").reset_index(drop=True)
        self.logger.info("Parsed %d CPI rows (%s .. %s)", len(df), df["Data"].min().date(), df["Data"].max().date())
        
        return df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    scraper = GusCpiScraper()
    df = scraper.run()
    print(df.tail())
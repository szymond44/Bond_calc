"""
run_pipeline.py

Single entrypoint for the whole pipeline -- this is what the GitHub Action
calls, and what you'd run locally to test. Each source is scraped,
validated, and published independently: if obligacjeskarbowe.pl's markup
breaks tomorrow, that must not stop CPI/NBP from updating, and a bad CPI
print must not get published just because bonds parsed fine.

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --only cpi,nbp
    python -m pipeline.run_pipeline --dry-run   # scrape + validate, don't publish
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

from pipeline.scrapers.base_scraper import ScraperError
from pipeline.scrapers.gus_cpi_scraper import GusCpiScraper
from pipeline.scrapers.nbp_rate_scraper import NbpRateScraper
from pipeline.scrapers.gov_pl_bonds_scraper import GovPlBondsScraper
from pipeline.validators.cpi_validator import validate_cpi
from pipeline.validators.nbp_validator import validate_nbp
from pipeline.validators.bonds_validator import validate_bonds
from pipeline.storage.publisher import (
    read_existing_csv,
    read_existing_bonds,
    publish_cpi,
    publish_nbp,
    publish_bonds,
)
from pipeline.config import CPI_CSV_PATH, NBP_CSV_PATH, BONDS_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("run_pipeline")


@dataclass
class SourceResult:
    name: str
    scraped: bool = False
    validated: bool = False
    published: bool = False
    messages: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        # Success for exit-code purposes means "validation passed" -- whether
        # it actually got published depends only on --dry-run, which is a
        # deliberate choice, not a failure.
        return self.validated

    def summary_line(self) -> str:
        if not self.scraped:
            status = "SCRAPE FAILED"
        elif not self.validated:
            status = "VALIDATION FAILED"
        elif self.published:
            status = "PUBLISHED"
        else:
            status = "VALIDATED (dry run, not published)"
        return f"[{status}] {self.name}"


def run_cpi(dry_run: bool) -> SourceResult:
    result = SourceResult(name="gus_cpi")
    try:
        new_df = GusCpiScraper().run()
        result.scraped = True
    except ScraperError as e:
        result.error = str(e)
        logger.error("CPI scrape failed: %s", e)
        return result

    existing_df = read_existing_csv(CPI_CSV_PATH)
    ok, messages = validate_cpi(new_df, existing_df)
    result.messages = messages
    result.validated = ok
    for msg in messages:
        (logger.info if ok else logger.warning)(msg)

    if ok and not dry_run:
        path = publish_cpi(new_df)
        result.published = True
        logger.info("Published CPI -> %s", path)
    return result


def run_nbp(dry_run: bool) -> SourceResult:
    result = SourceResult(name="nbp_rate")
    try:
        new_df = NbpRateScraper().run()
        result.scraped = True
    except ScraperError as e:
        result.error = str(e)
        logger.error("NBP scrape failed: %s", e)
        return result

    existing_df = read_existing_csv(NBP_CSV_PATH)
    ok, messages = validate_nbp(new_df, existing_df)
    result.messages = messages
    result.validated = ok
    for msg in messages:
        (logger.info if ok else logger.warning)(msg)

    if ok and not dry_run:
        path = publish_nbp(new_df)
        result.published = True
        logger.info("Published NBP rate -> %s", path)
    return result


def run_bonds(dry_run: bool) -> SourceResult:
    result = SourceResult(name="bonds")
    try:
        new_config = GovPlBondsScraper().run()
        result.scraped = True
    except ScraperError as e:
        result.error = str(e)
        logger.error("Bonds scrape failed: %s", e)
        return result

    existing_config = read_existing_bonds(BONDS_CONFIG_PATH)
    ok, messages = validate_bonds(new_config, existing_config)
    result.messages = messages
    result.validated = ok
    for msg in messages:
        (logger.info if ok else logger.warning)(msg)

    if ok and not dry_run:
        path = publish_bonds(new_config)
        result.published = True
        logger.info("Published bonds -> %s", path)
    return result


SOURCE_RUNNERS = {
    "cpi": run_cpi,
    "nbp": run_nbp,
    "bonds": run_bonds,
}


def main():
    parser = argparse.ArgumentParser(description="Scrape -> validate -> publish pipeline")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated subset of sources to run: {','.join(SOURCE_RUNNERS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and validate, but don't write to data/ or bonds_config.json",
    )
    args = parser.parse_args()

    sources = list(SOURCE_RUNNERS) if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = set(sources) - set(SOURCE_RUNNERS)
    if unknown:
        parser.error(f"Unknown source(s): {unknown}. Valid: {list(SOURCE_RUNNERS)}")

    results: list[SourceResult] = []
    for name in sources:
        logger.info("=== Running source: %s ===", name)
        results.append(SOURCE_RUNNERS[name](args.dry_run))

    logger.info("=== Pipeline summary ===")
    any_failed = False
    for r in results:
        logger.info(r.summary_line())
        if not r.ok:
            any_failed = True

    # Non-zero exit on any failure -- this is what makes the GitHub Action
    # step show red and (with the right workflow config) notify you, instead
    # of quietly committing a partial/stale update.
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()

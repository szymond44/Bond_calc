"""
base_scraper.py

Shared plumbing for all pipeline scrapers:
  - a requests.Session with sane timeouts + retry/backoff
  - consistent logging
  - writing a raw, untouched, timestamped snapshot BEFORE any transformation
    (see pipeline notes: if a transform bug is found later, you can re-derive
    corrected numbers from the original scrape instead of losing the source)

Every concrete scraper (GusCpiScraper, NbpRateScraper,
ObligacjeSkarboweScraper, ...) subclasses BaseScraper and implements run().
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.config import SNAPSHOT_ROOT

DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_USER_AGENT = (
    "BondSimulatorDataPipeline/0.1 "
    "(hobby project; contact: set-your-contact-email-here)"
)


class ScraperError(RuntimeError):
    """Raised when a scraper cannot produce usable data. Callers (run_pipeline.py)
    should catch this, log it, and skip publishing rather than crash the whole run."""


class BaseScraper(ABC):
    #: short machine-friendly name, e.g. "gus_cpi", "nbp_rate", "bonds"
    source_name: str = "base"

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 4,
        backoff_factor: float = 1.5,
        user_agent: str = DEFAULT_USER_AGENT,
        logger: Optional[logging.Logger] = None,
    ):
        self.timeout = timeout
        self.logger = logger or logging.getLogger(self.source_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            # Without this, messages also propagate up to the root logger
            # (e.g. the one run_pipeline.py sets up via basicConfig) and get
            # printed a second time by its handler -- every line doubled.
            self.logger.propagate = False

        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,  # 1.5s, 3s, 6s, 12s ...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _get(self, url: str, *, polite_delay: float = 0.5, **kwargs) -> requests.Response:
        """A single polite GET. `polite_delay` is a small fixed pause before
        the request -- cheap way to avoid hammering a source that updates
        monthly, not to avoid a rate limiter (Retry already handles 429s)."""
        if polite_delay:
            time.sleep(polite_delay)

        self.logger.info("GET %s", url)
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def fetch_json(self, url: str, **kwargs) -> Any:
        return self._get(url, **kwargs).json()

    def fetch_text(self, url: str, **kwargs) -> str:
        resp = self._get(url, **kwargs)
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def fetch_bytes(self, url: str, **kwargs) -> bytes:
        return self._get(url, **kwargs).content

    # ------------------------------------------------------------------
    # Snapshotting
    # ------------------------------------------------------------------
    def save_snapshot(self, raw_payload: Any, suffix: str = "raw") -> Path:
        """Writes the untouched scrape result to snapshots/<UTC date>/<source>_<suffix>.json
        with a timestamp + source URL trail. Called BEFORE validation/transformation.

        raw_payload can be a dict/list (json-serializable) or a string (e.g. raw XML/HTML) --
        strings are wrapped so the file is still valid JSON with metadata attached.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = SNAPSHOT_ROOT / today
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.source_name}_{suffix}.json"

        envelope = {
            "source": self.source_name,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": raw_payload,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info("Snapshot written: %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Contract every scraper implements
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self):
        """Fetch + snapshot + normalize. Must return a pandas.DataFrame (or similar
        tabular structure) ready to be handed to the matching validator in
        pipeline/validators/. Should NOT touch data/ directly -- that happens
        later, in storage/publisher.py, only after validation passes."""
        raise NotImplementedError

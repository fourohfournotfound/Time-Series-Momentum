"""Data provider clients for historical daily bars.

This module provides lightweight wrappers around the Polygon.io and
Alpaca Markets HTTP APIs.  The clients focus on retrieving adjusted
daily bars and returning them as pandas ``DataFrame`` instances that are
ready to use within the momentum backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class _ClientConfig:
    """Simple configuration container for HTTP clients."""

    timeout: int = 30
    max_retries: int = 5
    backoff_factor: float = 0.4


def _build_retry_adapter(config: _ClientConfig) -> HTTPAdapter:
    """Create a ``requests`` adapter with sensible retry defaults."""

    retry = Retry(
        total=config.max_retries,
        read=config.max_retries,
        connect=config.max_retries,
        backoff_factor=config.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry)


class BaseDataClient:
    """Base HTTP client with retry configuration."""

    def __init__(self, session: Optional[requests.Session] = None, *, config: Optional[_ClientConfig] = None) -> None:
        self.session = session or requests.Session()
        self.config = config or _ClientConfig()
        adapter = _build_retry_adapter(self.config)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def close(self) -> None:
        self.session.close()


class PolygonAggsClient(BaseDataClient):
    """Client for Polygon.io aggregate (daily) data."""

    base_url = "https://api.polygon.io"

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, *, config: Optional[_ClientConfig] = None) -> None:
        if not api_key:
            raise ValueError("A Polygon.io API key is required")
        super().__init__(session, config=config)
        self.api_key = api_key

    def get_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch adjusted daily aggregates for ``ticker``.

        Parameters
        ----------
        ticker
            Symbol to request.
        start, end
            Inclusive date range in ``YYYY-mm-dd`` format.
        """

        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params: Dict[str, str] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self.api_key,
        }
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])

        if not results:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n", "timestamp", "date"])

        frame = pd.DataFrame(results)
        frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        frame["ticker"] = ticker
        cols = ["t", "timestamp", "date", "ticker", "o", "h", "l", "c", "v", "vw", "n"]
        return frame[cols].sort_values("timestamp").reset_index(drop=True)


class AlpacaBarsClient(BaseDataClient):
    """Client for Alpaca Markets historical daily bars."""

    base_url = "https://data.alpaca.markets/v2"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        data_feed: str = "us",
        session: Optional[requests.Session] = None,
        config: Optional[_ClientConfig] = None,
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("Alpaca API and secret keys are required")
        super().__init__(session, config=config)
        self.session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        })
        self.data_feed = data_feed

    def get_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch adjusted daily bars for ``ticker`` from Alpaca."""

        params: Dict[str, str] = {
            "timeframe": "1Day",
            "adjustment": "all",
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "limit": "10000",
            "feed": self.data_feed,
        }

        url = f"{self.base_url}/stocks/{ticker}/bars"
        page_token: Optional[str] = None

        frames = []

        while True:
            if page_token:
                params["page_token"] = page_token
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            payload = response.json()
            bars = payload.get("bars", [])
            if not bars:
                break
            frame = pd.DataFrame(bars)
            frames.append(frame)
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if not frames:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n", "timestamp", "date"])

        frame = pd.concat(frames, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert("America/New_York")
        frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        frame["ticker"] = ticker
        cols = ["t", "timestamp", "date", "ticker", "o", "h", "l", "c", "v", "vw", "n"]
        return frame[cols].sort_values("timestamp").reset_index(drop=True)


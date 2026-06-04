"""FMP data loader → NautilusTrader Bar objects."""

from __future__ import annotations

import os
from datetime import timezone

import pandas as pd
import requests

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity


FMP_BASE = "https://financialmodelingprep.com/stable"


def fetch_fmp_daily(
    symbol: str,
    start: str = "2000-01-01",
    end: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV from FMP and return a sorted DataFrame."""
    key = api_key or os.environ.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError("FMP_API_KEY not set")

    url = f"{FMP_BASE}/historical-price-eod/full"
    params: dict[str, str] = {"symbol": symbol, "from": start, "apikey": key}
    if end:
        params["to"] = end

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if not data or not isinstance(data, list):
        raise ValueError(f"No historical data for {symbol}: {data}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def dataframe_to_bars(
    df: pd.DataFrame,
    bar_type: BarType,
    price_precision: int = 2,
) -> list[Bar]:
    """Convert a FMP DataFrame to a list of NautilusTrader Bar objects."""
    bars: list[Bar] = []
    for _, row in df.iterrows():
        ts_ns = int(
            row["date"]
            .to_pydatetime()
            .replace(hour=16, tzinfo=timezone.utc)
            .timestamp()
            * 1e9
        )
        bar = Bar(
            bar_type=bar_type,
            open=Price(row["open"], price_precision),
            high=Price(row["high"], price_precision),
            low=Price(row["low"], price_precision),
            close=Price(row["close"], price_precision),
            volume=Quantity(float(row["volume"]), 0),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    return bars


def load_spy_bars(
    bar_type_str: str = "SPY.XNAS-1-DAY-LAST-EXTERNAL",
    start: str = "2000-01-01",
    end: str | None = None,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, list[Bar]]:
    """Convenience: fetch SPY from FMP, return (df, bars)."""
    df = fetch_fmp_daily("SPY", start=start, end=end, api_key=api_key)
    bar_type = BarType.from_str(bar_type_str)
    bars = dataframe_to_bars(df, bar_type)
    return df, bars

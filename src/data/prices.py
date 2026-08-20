"""Yahoo Finance price history download + on-disk cache.

Produces a tidy long-format DataFrame:
    date | ticker | open | high | low | close | volume

`close` is split- and dividend-adjusted (auto_adjust=True), which is what you
want for momentum/volatility maths -- raw closes create fake -50% days on
split dates.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..config import CACHE_DIR, Config

_PRICE_CACHE = CACHE_DIR / "prices.parquet"


def _cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < max_age_hours


def _flatten(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance returns MultiIndex columns for multi-ticker downloads.

    With group_by="ticker" the levels are (ticker, field). We stack the ticker
    level into rows so downstream code never has to think about MultiIndexes.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(0):
                continue
            sub = raw[tkr].copy()
            sub["ticker"] = tkr
            frames.append(sub)
        out = pd.concat(frames)
    else:  # single ticker -> flat columns
        out = raw.copy()
        out["ticker"] = tickers[0]

    out = out.reset_index()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "index" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"index": "date"})

    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    out = out[[c for c in keep if c in out.columns]]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.dropna(subset=["close"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Download (or load from cache) the full price history for the universe."""
    tickers = list(dict.fromkeys(cfg.universe + [cfg.benchmark]))
    max_age = float(cfg.data.get("cache_max_age_hours", 12))

    if not force and _cache_is_fresh(_PRICE_CACHE, max_age):
        cached = pd.read_parquet(_PRICE_CACHE)
        if set(tickers).issubset(set(cached["ticker"].unique())):
            print(f"[prices] cache hit: {_PRICE_CACHE.name} "
                  f"({len(cached):,} rows, {cached['ticker'].nunique()} tickers)")
            return cached

    print(f"[prices] downloading {len(tickers)} tickers, period={cfg.data['period']} "
          f"from Yahoo Finance ...")
    raw = yf.download(
        tickers=tickers,
        period=cfg.data["period"],
        interval=cfg.data["interval"],
        auto_adjust=True,     # adjust for splits AND dividends
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data -- check network/tickers.")

    df = _flatten(raw, tickers)

    # Report per-ticker coverage so a silently-truncated ticker is visible.
    span = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    short = span[span["count"] < 252 * 10]
    if not short.empty:
        print(f"[prices] WARNING: {len(short)} ticker(s) have <10y of daily bars:")
        for tkr, row in short.iterrows():
            print(f"          {tkr}: {row['count']} bars "
                  f"({row['min'].date()} -> {row['max'].date()})")

    df.to_parquet(_PRICE_CACHE, index=False)
    print(f"[prices] saved {len(df):,} rows -> {_PRICE_CACHE} "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    return df


def latest_closes(prices: pd.DataFrame) -> dict[str, float]:
    """Most recent close per ticker -- used to mark the portfolio to market."""
    last = prices.sort_values("date").groupby("ticker").tail(1)
    return dict(zip(last["ticker"], last["close"].astype(float)))

"""Realised profit-and-loss accounting.

Two distinct things live here, and conflating them is the main trap:

1. `annual_returns()` -- calendar-year total return of each STOCK, straight
   from the price history. This is what the market did. It is available from
   the first run because it needs no portfolio history at all.

2. `portfolio_annual_pnl()` -- calendar-year profit of THIS PORTFOLIO, derived
   from valuation snapshots recorded on each `--apply` run. This is what the
   agent actually earned, and it only becomes meaningful once the portfolio has
   been carried across multiple runs.

Neither is a backtest. See the README -- a backtest would have to replay
history point-in-time and re-ask the model at each rebalance date. These
numbers describe outcomes; they do not attribute them to skill.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

# How many calendar years of per-stock history to surface. Ten keeps the
# prompt affordable while covering a full cycle.
DEFAULT_YEARS = 10


def annual_returns(prices: pd.DataFrame, tickers: list[str],
                   years: int = DEFAULT_YEARS) -> pd.DataFrame:
    """Calendar-year total return (%) per ticker.

    Uses adjusted closes, so these are total returns including dividends. The
    current year is a partial year-to-date figure -- it is labelled as such by
    `current_year_is_partial()` rather than silently compared to full years.
    """
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    cols = [t for t in tickers if t in wide.columns]
    wide = wide[cols]

    # Last observation of each calendar year, then year-over-year change.
    year_end = wide.groupby(wide.index.year).last()
    # Seed with the prior year's close so the first retained year is a real
    # return rather than NaN.
    pct = year_end.pct_change() * 100.0
    pct = pct.tail(years)
    pct.index = pct.index.astype(int)
    return pct.round(2)


def current_year_is_partial(prices: pd.DataFrame) -> bool:
    last = pd.to_datetime(prices["date"]).max()
    return not (last.month == 12 and last.day >= 29)


def equal_weight_baseline(annual: pd.DataFrame) -> pd.Series:
    """Return of an equally-weighted, annually-rebalanced basket of the universe.

    The cross-sectional mean of per-stock annual returns is exactly the return
    of a basket rebalanced to equal weights at each year end. This is the
    honest 'no skill' comparison: it is what you would have earned by buying
    the whole watch list and thinking about nothing.
    """
    return annual.mean(axis=1).round(2)


def summarise_stock_performance(annual: pd.DataFrame, benchmark_col: pd.Series | None
                                ) -> dict:
    """Compact per-year table for the report and the prompt."""
    baseline = equal_weight_baseline(annual)
    rows = []
    for year in annual.index:
        row = {
            "year": int(year),
            "equal_weight_universe_pct": _f(baseline.get(year)),
        }
        if benchmark_col is not None and year in benchmark_col.index:
            row["benchmark_pct"] = _f(benchmark_col.get(year))
        best = annual.loc[year].idxmax()
        worst = annual.loc[year].idxmin()
        row["best"] = f"{best} {annual.loc[year, best]:+.1f}%"
        row["worst"] = f"{worst} {annual.loc[year, worst]:+.1f}%"
        rows.append(row)
    return {"by_year": rows,
            "per_ticker": {t: {int(y): _f(annual.loc[y, t]) for y in annual.index}
                           for t in annual.columns}}


def per_ticker_consistency(annual: pd.DataFrame) -> dict[str, dict]:
    """Hit rate and best/worst year per ticker.

    Given to the model so it can tell a steady compounder from a stock whose
    ten-year average was manufactured by one explosive year.
    """
    out = {}
    for ticker in annual.columns:
        series = annual[ticker].dropna()
        if series.empty:
            continue
        out[ticker] = {
            "positive_years": int((series > 0).sum()),
            "total_years": int(len(series)),
            "best_year_pct": _f(series.max()),
            "worst_year_pct": _f(series.min()),
            "median_year_pct": _f(series.median()),
        }
    return out


# --------------------------------------------------------------------------
# portfolio-level P&L, from recorded snapshots
# --------------------------------------------------------------------------
def portfolio_annual_pnl(state: dict) -> dict:
    """Calendar-year profit of the tracked portfolio.

    Derived from the `history` snapshots written by `record_snapshot()`. There
    are no deposits or withdrawals in this simulation, so profit for a year is
    simply (last value in year) - (last value in the prior year), with the
    first year measured from the starting cash.

    Returns a dict with `available: False` when there is not enough history --
    the report then says so instead of printing a fabricated zero.
    """
    history = state.get("history", [])
    if len(history) < 2:
        return {
            "available": False,
            "reason": (f"only {len(history)} portfolio snapshot(s) recorded. "
                       "Profit-per-year needs at least two runs persisted with "
                       "--apply."),
            "snapshots": len(history),
        }

    snaps = sorted(history, key=lambda s: s["date"])
    by_year: dict[int, list] = {}
    for snap in snaps:
        year = int(str(snap["date"])[:4])
        by_year.setdefault(year, []).append(snap)

    inception_value = float(snaps[0]["total_value"])
    rows = []
    prev_close = inception_value
    for year in sorted(by_year):
        last = by_year[year][-1]
        end_value = float(last["total_value"])
        # The first year opens at inception, not at the prior year's close.
        start_value = prev_close
        profit = end_value - start_value
        rows.append({
            "year": year,
            "start_value": round(start_value, 2),
            "end_value": round(end_value, 2),
            "profit_usd": round(profit, 2),
            "profit_pct": round(profit / start_value * 100, 2) if start_value else 0.0,
            "snapshots": len(by_year[year]),
            "realised_pnl_usd": round(sum(
                float(s.get("realised_pnl_in_run", 0.0)) for s in by_year[year]), 2),
        })
        prev_close = end_value

    total_profit = prev_close - inception_value
    return {
        "available": True,
        "inception_date": snaps[0]["date"],
        "inception_value": round(inception_value, 2),
        "current_value": round(prev_close, 2),
        "total_profit_usd": round(total_profit, 2),
        "total_profit_pct": round(total_profit / inception_value * 100, 2)
        if inception_value else 0.0,
        "total_realised_pnl_usd": round(
            sum(float(s.get("realised_pnl_in_run", 0.0)) for s in snaps), 2),
        "by_year": rows,
        "snapshots": len(snaps),
    }


def _f(x) -> float | None:
    if x is None or pd.isna(x):
        return None
    return round(float(x), 2)

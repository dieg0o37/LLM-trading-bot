"""Pre-computed technical indicators.

The LLM never sees raw OHLCV -- 15 tickers x 15 years of daily bars is ~56k
rows and would blow the context window while telling the model very little.
Instead pandas reduces each ticker to ~25 scalar metrics that a human PM would
actually look at. This is the single most important design choice in the
project: the model does judgement, pandas does arithmetic.

Every metric is computed as of the LAST available bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# individual indicator helpers (operate on a single ticker's close series)
# --------------------------------------------------------------------------
def _total_return(close: pd.Series, lookback: int) -> float | None:
    """Simple total return over `lookback` trading days."""
    if len(close) <= lookback:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - lookback] - 1.0)


def _momentum_12_1(close: pd.Series) -> float | None:
    """Classic academic momentum: 12-month return SKIPPING the most recent month.

    The last month is excluded because short-term reversal contaminates raw
    12m momentum -- this is the Jegadeesh-Titman construction.
    """
    if len(close) <= TRADING_DAYS:
        return None
    return float(close.iloc[-21] / close.iloc[-TRADING_DAYS] - 1.0)


def _annualised_vol(returns: pd.Series, window: int) -> float | None:
    if len(returns) < window:
        return None
    return float(returns.iloc[-window:].std(ddof=1) * np.sqrt(TRADING_DAYS))


def _max_drawdown(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    w = close.iloc[-window:]
    return float((w / w.cummax() - 1.0).min())


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI (EMA-smoothed, alpha = 1/period)."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float | None:
    """Average True Range as a % of price -- a scale-free volatility read."""
    if len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(atr / close.iloc[-1])


def _beta_and_corr(returns: pd.Series, bench: pd.Series, window: int = TRADING_DAYS * 2
                   ) -> tuple[float | None, float | None]:
    """OLS beta and correlation vs the benchmark over a 2-year daily window."""
    joined = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return None, None
    joined = joined.iloc[-window:]
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = b.var(ddof=1)
    if var_b == 0:
        return None, None
    beta = float(r.cov(b) / var_b)
    return beta, float(r.corr(b))


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def compute_metrics(prices: pd.DataFrame, universe: list[str], benchmark: str) -> pd.DataFrame:
    """One row per ticker, ~25 columns of pre-computed metrics."""
    wide_close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    daily_returns = wide_close.pct_change()

    if benchmark not in daily_returns.columns:
        raise ValueError(f"benchmark {benchmark} missing from price data")
    bench_returns = daily_returns[benchmark]

    rows = []
    for ticker in universe:
        sub = prices[prices["ticker"] == ticker].sort_values("date")
        if sub.empty:
            print(f"[technicals] WARNING: no price data for {ticker}, skipped")
            continue

        close = sub["close"].reset_index(drop=True)
        rets = close.pct_change().dropna()
        px = float(close.iloc[-1])

        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        window_52w = close.iloc[-TRADING_DAYS:]
        high_52w, low_52w = float(window_52w.max()), float(window_52w.min())

        vol_20 = float(sub["volume"].iloc[-20:].mean()) if len(sub) >= 20 else None
        vol_90 = float(sub["volume"].iloc[-90:].mean()) if len(sub) >= 90 else None

        beta, corr = _beta_and_corr(daily_returns[ticker].dropna(), bench_returns)
        vol_1y = _annualised_vol(rets, TRADING_DAYS)
        ret_1y = _total_return(close, TRADING_DAYS)

        rows.append({
            "ticker": ticker,
            "price": round(px, 2),
            "history_start": sub["date"].min().date().isoformat(),
            "history_years": round(len(close) / TRADING_DAYS, 1),

            # --- returns -------------------------------------------------
            "ret_1w_pct":  _pct(_total_return(close, 5)),
            "ret_1m_pct":  _pct(_total_return(close, 21)),
            "ret_3m_pct":  _pct(_total_return(close, 63)),
            "ret_6m_pct":  _pct(_total_return(close, 126)),
            "ret_12m_pct": _pct(ret_1y),
            "ret_3y_ann_pct": _pct(_annualised(close, TRADING_DAYS * 3)),
            "ret_10y_ann_pct": _pct(_annualised(close, TRADING_DAYS * 10)),
            "momentum_12_1_pct": _pct(_momentum_12_1(close)),

            # --- risk ----------------------------------------------------
            "vol_30d_ann_pct": _pct(_annualised_vol(rets, 30)),
            "vol_90d_ann_pct": _pct(_annualised_vol(rets, 90)),
            "vol_1y_ann_pct":  _pct(vol_1y),
            "max_drawdown_1y_pct": _pct(_max_drawdown(close, TRADING_DAYS)),
            "max_drawdown_3y_pct": _pct(_max_drawdown(close, TRADING_DAYS * 3)),
            "atr_14_pct": _pct(_atr_pct(sub)),
            "beta_vs_bench": _round(beta, 2),
            "corr_vs_bench": _round(corr, 2),
            # Return per unit of risk. Not a true Sharpe (no risk-free rate),
            # labelled honestly so the model does not over-read it.
            "return_over_vol_1y": _round(ret_1y / vol_1y if (ret_1y and vol_1y) else None, 2),

            # --- trend / location ----------------------------------------
            "sma_50": _round(sma50, 2),
            "sma_200": _round(sma200, 2),
            "px_vs_sma50_pct": _pct(px / sma50 - 1 if sma50 else None),
            "px_vs_sma200_pct": _pct(px / sma200 - 1 if sma200 else None),
            "golden_cross": bool(sma50 > sma200) if (sma50 and sma200) else None,
            "rsi_14": _round(_rsi(close), 1),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "pct_below_52w_high": _pct(px / high_52w - 1),
            "pct_above_52w_low": _pct(px / low_52w - 1),

            # --- participation -------------------------------------------
            "volume_20d_vs_90d": _round(vol_20 / vol_90 if (vol_20 and vol_90) else None, 2),
        })

    return pd.DataFrame(rows).set_index("ticker", drop=False)


def _annualised(close: pd.Series, lookback: int) -> float | None:
    """CAGR over `lookback` trading days."""
    if len(close) <= lookback:
        return None
    total = close.iloc[-1] / close.iloc[-1 - lookback]
    years = lookback / TRADING_DAYS
    return float(total ** (1 / years) - 1.0)


def _pct(x: float | None, nd: int = 2) -> float | None:
    return None if x is None or not np.isfinite(x) else round(x * 100.0, nd)


def _round(x: float | None, nd: int) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def correlation_matrix(prices: pd.DataFrame, universe: list[str], window: int = TRADING_DAYS
                       ) -> pd.DataFrame:
    """Pairwise return correlation over the last year.

    Given to the model so "diversify" is a data-backed instruction rather than
    a vibe -- it can see that MSFT/NVDA move together and XOM does not.
    """
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    cols = [t for t in universe if t in wide.columns]
    rets = wide[cols].pct_change().iloc[-window:]
    return rets.corr().round(2)

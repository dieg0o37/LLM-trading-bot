"""Builds the user-message payload the model reasons over.

Kept separate from the API call so the exact JSON sent to Claude can be dumped
to disk and inspected -- `run.py --dry-run` writes it without spending a token.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd


def build_payload(valuation: dict, metrics: pd.DataFrame, corr: pd.DataFrame,
                  news: dict, cfg) -> dict:
    """Assemble every fact the model is allowed to use."""
    # Drop all-null columns so the model is not handed a wall of `null`s.
    records = json.loads(metrics.drop(columns=["ticker"]).reset_index().to_json(
        orient="records"))

    held = {h["ticker"] for h in valuation["holdings"]}
    for rec in records:
        rec["currently_held"] = rec["ticker"] in held
        if news.get(rec["ticker"]):
            n = news[rec["ticker"]]
            rec["news"] = {
                "article_count": n["article_count"],
                "mean_sentiment": n["mean_sentiment"],
                "headlines": [
                    {"title": h["title"], "sentiment": h["sentiment_label"],
                     "published": h["published"]}
                    for h in n["headlines"]
                ],
            }

    return {
        "as_of": date.today().isoformat(),
        "constraints": {
            "max_single_position_pct": cfg.max_position_pct,
            "min_cash_pct": cfg.min_cash_pct,
            "whole_shares_only": True,
            "tradable_tickers": cfg.universe,
        },
        "current_portfolio": {
            "total_value": valuation["total_value"],
            "cash": valuation["cash"],
            "cash_pct": valuation["cash_pct"],
            "holdings": valuation["holdings"],
        },
        "watchlist_metrics": records,
        "correlation_matrix_1y": json.loads(corr.to_json(orient="index")),
        "news_available": bool(news),
    }


def render_user_message(payload: dict) -> str:
    """The volatile half of the request -- everything after the cache breakpoint."""
    holdings = payload["current_portfolio"]["holdings"]
    position_line = (
        ", ".join(f"{h['ticker']} {h['weight_pct']:.1f}%" for h in holdings)
        if holdings else "none -- the portfolio is 100% cash"
    )
    news_line = ("Recent news sentiment is included per ticker."
                 if payload["news_available"]
                 else "No news data is available this run; decide on price and "
                      "technical metrics alone.")

    return (
        f"Portfolio review for {payload['as_of']}.\n\n"
        f"Total value ${payload['current_portfolio']['total_value']:,.2f} | "
        f"cash {payload['current_portfolio']['cash_pct']:.1f}% | "
        f"positions: {position_line}\n\n"
        f"{news_line}\n\n"
        f"Full data follows as JSON.\n\n"
        f"<portfolio_data>\n{json.dumps(payload, indent=1)}\n</portfolio_data>\n\n"
        "Review every watch-list ticker against the current positions, then call "
        "submit_rebalance_plan exactly once with your decision."
    )

"""Alpha Vantage NEWS_SENTIMENT client.

Schema notes (verified against a live response, 2026-08-20):
  top level : items (str!), sentiment_score_definition, relevance_score_definition, feed
  feed[]    : title, url, time_published, authors, summary, banner_image, source,
              category_within_source, source_domain, topics[],
              overall_sentiment_score (float), overall_sentiment_label (str),
              ticker_sentiment[]
  ticker_sentiment[] : ticker, relevance_score, ticker_sentiment_score,
              ticker_sentiment_label   <- ALL STRINGS, must be float()-cast

Gotchas this module handles:
  * Errors come back as HTTP 200 with an {"Information": ...} or {"Note": ...}
    body, so `raise_for_status()` alone never fires.
  * `time_published` is YYYYMMDDTHHMMSS (15 chars) on the way OUT, while the
    `time_from` request parameter is YYYYMMDDTHHMM (13 chars). Different.
  * Free tier is 25 requests/day, account-wide. We issue one request per
    ticker (`tickers=` is AND-ed, so a combined call would only return
    articles mentioning every ticker at once) and cache hard.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ..config import CACHE_DIR, Config

_ENDPOINT = "https://www.alphavantage.co/query"
_NEWS_CACHE = CACHE_DIR / "news"
_TIMEOUT = 30


class AlphaVantageError(RuntimeError):
    """Raised when the API returns an error body (which it does with HTTP 200)."""


def _cache_path(ticker: str) -> Path:
    return _NEWS_CACHE / f"{ticker}.json"


def _fetch_one(ticker: str, api_key: str, lookback_days: int, limit: int) -> dict:
    time_from = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y%m%dT%H%M")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "sort": "LATEST",
        "limit": str(max(limit * 4, 50)),  # over-fetch, then filter by relevance
        "apikey": api_key,
    }
    resp = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    # Alpha Vantage signals every error class with HTTP 200 + a message key.
    for key in ("Information", "Note", "Error Message"):
        if key in payload:
            raise AlphaVantageError(f"{key}: {payload[key]}")
    if "feed" not in payload:
        raise AlphaVantageError(f"unexpected payload keys: {sorted(payload)}")
    return payload


def _summarise(ticker: str, payload: dict, top_n: int) -> dict:
    """Collapse a raw NEWS_SENTIMENT payload into the compact shape the LLM sees."""
    articles = []
    for item in payload.get("feed", []):
        # Pull out THIS ticker's sentiment; the article-level score covers all
        # mentioned tickers and is much noisier.
        ts = next((t for t in item.get("ticker_sentiment", [])
                   if t.get("ticker") == ticker), None)
        if ts is None:
            continue
        try:
            relevance = float(ts["relevance_score"])
            score = float(ts["ticker_sentiment_score"])
        except (KeyError, TypeError, ValueError):
            continue

        raw_time = item.get("time_published", "")
        try:
            published = datetime.strptime(raw_time, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
        except ValueError:
            published = raw_time[:8]

        articles.append({
            "title": item.get("title", "")[:180],
            "source": item.get("source", ""),
            "published": published,
            "relevance": round(relevance, 3),
            "sentiment_score": round(score, 3),
            "sentiment_label": ts.get("ticker_sentiment_label", ""),
        })

    # Rank by relevance to this ticker, keep the top N for the prompt.
    articles.sort(key=lambda a: a["relevance"], reverse=True)
    kept = articles[:top_n]

    mean_sent = round(sum(a["sentiment_score"] for a in articles) / len(articles), 3) if articles else None
    return {
        "ticker": ticker,
        "article_count": len(articles),
        "mean_sentiment": mean_sent,
        "headlines": kept,
    }


def download_news(cfg: Config, force: bool = False) -> dict[str, dict]:
    """Return {ticker: summary}. Degrades gracefully -- never raises upward.

    A missing key, a blown quota, or a network failure all produce an empty
    dict and a printed reason; the pipeline then runs price-only.
    """
    if not cfg.news_enabled:
        reason = ("no ALPHAVANTAGE_API_KEY in .env" if not cfg.alphavantage_api_key
                  else "news.enabled = false in config.yaml")
        print(f"[news] skipped -- {reason}. Running price/technicals-only.")
        return {}

    _NEWS_CACHE.mkdir(parents=True, exist_ok=True)
    max_age = float(cfg.news.get("cache_max_age_hours", 24))
    top_n = int(cfg.news.get("articles_per_ticker", 5))
    lookback = int(cfg.news.get("lookback_days", 7))

    out: dict[str, dict] = {}
    fetched = quota_hit = 0

    for ticker in cfg.universe:
        path = _cache_path(ticker)
        fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 < max_age

        if fresh and not force:
            payload = json.loads(path.read_text())
        elif quota_hit:
            # Quota is account-wide -- once we hit it, stop hammering and fall
            # back to whatever stale cache exists for the remaining tickers.
            if path.exists():
                payload = json.loads(path.read_text())
                print(f"[news] {ticker}: quota exhausted, using stale cache")
            else:
                continue
        else:
            try:
                payload = _fetch_one(ticker, cfg.alphavantage_api_key, lookback, top_n)
                path.write_text(json.dumps(payload))
                fetched += 1
                time.sleep(1.0)  # be polite; free tier has no documented rpm cap
            except AlphaVantageError as exc:
                print(f"[news] {ticker}: {exc}")
                quota_hit = 1
                if path.exists():
                    payload = json.loads(path.read_text())
                else:
                    continue
            except requests.RequestException as exc:
                print(f"[news] {ticker}: network error ({exc.__class__.__name__}) -- skipping")
                continue

        out[ticker] = _summarise(ticker, payload, top_n)

    print(f"[news] {len(out)}/{len(cfg.universe)} tickers with headlines "
          f"({fetched} fresh API calls, rest from cache)")
    return out

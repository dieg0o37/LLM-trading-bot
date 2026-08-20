"""Configuration loading: config.yaml for settings, .env for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
REPORT_DIR = ROOT / "data" / "reports"
PORTFOLIO_PATH = ROOT / "portfolio.json"


@dataclass
class Config:
    universe: list[str]
    benchmark: str
    data: dict[str, Any]
    news: dict[str, Any]
    portfolio: dict[str, Any]
    llm: dict[str, Any]
    anthropic_api_key: str | None = None
    alphavantage_api_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def max_position_pct(self) -> float:
        return float(self.portfolio["max_position_pct"])

    @property
    def min_cash_pct(self) -> float:
        return float(self.portfolio["min_cash_pct"])

    @property
    def news_enabled(self) -> bool:
        """News needs both the config flag and an actual key."""
        return bool(self.news.get("enabled")) and bool(self.alphavantage_api_key)


def load_config(path: Path | str = ROOT / "config.yaml") -> Config:
    load_dotenv(ROOT / ".env")
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    return Config(
        universe=raw["universe"],
        benchmark=raw["benchmark"],
        data=raw["data"],
        news=raw["news"],
        portfolio=raw["portfolio"],
        llm=raw["llm"],
        # Note: we never print these. The SDK also reads ANTHROPIC_API_KEY on
        # its own -- we read it here only so we can fail early with a clear
        # message instead of a 401 halfway through the run.
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY") or None,
        raw=raw,
    )

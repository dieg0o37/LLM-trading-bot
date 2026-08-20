# LLM Trading Prog — an LLM as Head Portfolio Manager

A **notification-only** portfolio agent. It downloads 15 years of prices,
reduces them to ~30 technical metrics per stock with pandas, attaches news
sentiment, and asks **Claude Haiku 4.5** to act as a Head Portfolio Manager
under two hard constraints: no position above 25%, cash never below 10%.

**The goal is to see the Anthropic API in action** — system prompts, forced
strict tool use, structured outputs, prompt caching, usage telemetry — with
portfolio management as a domain that makes those mechanics legible. It is not
a trading bot and not investment advice. No broker is connected; the only
outputs are a console printout and files in `data/reports/`.

---

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Add `ANTHROPIC_API_KEY` (and optionally `ALPHAVANTAGE_API_KEY`), then:

```bash
python3 run.py
```

`--dry-run` builds the prompt without calling the API. `python3 -m tests.test_offline`
runs 54 assertions against a mocked response.

---

## Architecture

```
Yahoo Finance ─▶ prices.py ─────▶ technicals.py ──┐
(15y OHLCV,      parquet cache    ~30 metrics     │
 60k rows)                        + 15×15 corr    │
                                  performance.py  ├─▶ payload.py ─▶ manager.py
Alpha Vantage ─▶ news.py ────────▶ per-ticker     │   one JSON      claude-haiku-4-5
NEWS_SENTIMENT   24h cache         sentiment      │   payload       forced tool use
                                                  │                      │
portfolio.json ─▶ portfolio.py ───────────────────┘                      ▼
                  positions, cash                     portfolio.py ◀── plan JSON
                                                      simulate fills,
                                                      VERIFY constraints
                                                             │
                                                             ▼
                                                       report.py
                                                  console + .json + .md
```

| Path | Responsibility |
|---|---|
| `run.py` | CLI entry point; wires the pipeline together |
| `config.yaml` / `src/config.py` | Universe, windows, constraints, LLM settings |
| `src/data/prices.py` | yfinance download, MultiIndex flattening, parquet cache |
| `src/data/news.py` | Alpha Vantage client, caching, graceful degradation |
| `src/features/technicals.py` | All pandas indicator maths |
| `src/features/performance.py` | Calendar-year returns, equal-weight baseline, portfolio P&L |
| `src/agent/prompts.py` | System prompt + the JSON schema for the plan |
| `src/agent/payload.py` | Builds and renders the user message |
| `src/agent/manager.py` | The Anthropic API call, both output modes, error handling |
| `src/portfolio.py` | State, mark-to-market, fills, realised-P&L ledger, **constraint verifier** |
| `src/report.py` | Console, JSON, and Markdown output |
| `tests/test_offline.py` | 54-assertion end-to-end test, API mocked |

---

## The Anthropic API surface

All in `src/agent/manager.py` and `src/agent/prompts.py`.

**Model** — `claude-haiku-4-5`. It predates adaptive thinking, so extended
thinking uses the older `{"type": "enabled", "budget_tokens": N}` form, and
`output_config.effort` is unsupported and never sent.

**System prompt** — the PM brief plus seven operating rules and a **metric
glossary**. The glossary is load-bearing: without it the model must guess
whether `momentum_12_1_pct` is a price, a ratio, or a percent. Rule 6 requires
every rationale to cite a specific metric value; rule 7 forbids appeals to
knowledge outside the payload.

**Forced strict tool use** (default) —

```python
tools=[PLAN_TOOL]                     # "strict": True
tool_choice={"type": "tool", "name": "submit_rebalance_plan"}
```

`strict: True` with `additionalProperties: false` guarantees `tool_use.input`
validates exactly — `action` confined to `BUY|ADD|TRIM|SELL`,
`target_weight_pct` bounded to 0–25. `block.input` is read as parsed JSON,
never string-matched. The schema deliberately has **no share-count field**; see
Results.

**Structured outputs** (alternative) — `--output-mode json_schema` sends the
*same* `PLAN_SCHEMA` through `output_config.format`. Both paths are driven from
one constant and tested to agree.

**Prompt caching** — the system prompt carries `cache_control: {"type": "ephemeral"}`
and `SYSTEM_PROMPT` is frozen (no dates, no values) so the prefix stays stable.
It did not engage in practice — see Results.

**Error handling** — a most-specific-first chain (`AuthenticationError` →
`NotFoundError` → `RateLimitError` → `APIStatusError` → `APIConnectionError`),
each mapped to an actionable message.

**Telemetry** — every run reports input/output tokens, both cache counters,
`stop_reason`, message id, and estimated cost.

---

## What the model is given

pandas does the arithmetic; the model does the judgement. Raw OHLCV (15 tickers
× 15 years ≈ 56k rows) never reaches the prompt. Each ticker is reduced to
scalars as of the last bar:

- **Returns** — 1w/1m/3m/6m/12m, 3y and 10y CAGR, and `momentum_12_1_pct`
  (12-month return skipping the last month — the Jegadeesh–Titman construction).
- **Risk** — annualised vol at 30d/90d/1y, max drawdown 1y/3y, ATR(14) as % of
  price, beta and correlation vs SPY, and `return_over_vol_1y` (deliberately
  *not* called "Sharpe" — no risk-free rate is subtracted).
- **Trend** — SMA 50/200 and distance from them, `golden_cross`, RSI(14), 52-week
  range position.
- **Participation** — 20d vs 90d average volume.
- **Cross-sectional** — a 15×15 one-year correlation matrix, so "diversify" is
  data-backed rather than a vibe.
- **Calendar-year** — 10 years of per-year returns plus `annual_consistency`
  (positive years out of total, best/worst/median), to separate a steady
  compounder from one explosive year.
- **News** — top 5 headlines per ticker by relevance, with per-ticker sentiment.

Universe: AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, JNJ, PG, WMT, UNH, HD,
KO, CAT. SPY is downloaded as the beta benchmark but is never tradable.

**Data notes.** Closes are `auto_adjust=True` (split *and* dividend adjusted) —
raw closes create fake −50% days on split dates. Alpha Vantage returns errors as
**HTTP 200** with an `Information` key, `time_published` (15 chars) differs from
the `time_from` input format (13 chars), and `ticker_sentiment` scores are
strings while `overall_sentiment_score` is a float — all handled. The free tier
is 25 requests/day and `tickers=` is AND-ed, so the client issues one call per
ticker (15/day) with 24h caching, degrading to price-only on any failure.

---

## Constraints: stated to the model, verified in Python

The two rules are enforced twice — declared in the system prompt, then
**re-derived from scratch** after the model responds. **Only the Python result
is trusted.** The report prints the model's self-check beside the Python verdict
so disagreement is visible. Violations exit `2` and block `--apply`.

Python also owns the sizing. The model emits `target_weight_pct` and nothing
else; `apply_plan()` converts it:

```python
target_shares = floor(target_weight_pct / 100 * total_value / price)
```

`total_value` comes from the pre-trade portfolio and is held fixed — that is
exact, not an approximation, since buying converts cash into stock of equal
market value. Flooring biases every position slightly *below* target, so
rounding can only free cash, never consume more than intended.

*Model proposes, code disposes* — the most transferable pattern here, and the
Results below are why.

---

## Results

Three live runs against `claude-haiku-4-5`, same data and prompt. Runs 1–2 let
the model emit share counts; run 3 is after the fix, with the model supplying
target weights only.

| | Run 1 | Run 2 | Run 3 (after fix) |
|---|---|---|---|
| Verdict | **FAILED** — cash 9.34% | PASSED | PASSED |
| Predicted cash | 10.1% | 10.0% | 28% |
| Actual cash | 9.34% | **34.87%** | **29.67%** |
| Mean \|target − actual\| | 0.1 pp | **3.9 pp** | **0.24 pp** |
| Worst single error | +0.1 pp | AAPL **+13.7 pp** | CAT −0.76 pp |
| Tokens in/out · cost | 21,287 / 2,456 · $0.0336 | 21,287 / 1,731 · $0.0299 | 21,432 / 1,597 · $0.0294 |

> These are outcomes from three runs, not a backtest. Nothing here shows the agent's
> decisions beat the equal-weight universe or SPY — the report prints both columns so
> the comparison stays visible.

## Output

Three artefacts per run, timestamped in `data/reports/`: the **console**
printout (assessment, action table, fills, before/after weights, constraint
verdict, profit-per-year, API telemetry), a **`.json`** with the complete run,
and a **`.md`** human-readable report.

`portfolio.json` holds cash, positions, a trade ledger with realised P&L, and
valuation snapshots. Runs are **read-only by default** — `--apply` opts into
persistence and records a snapshot, which is what feeds profit-per-year.

| Flag | Effect |
|---|---|
| `--dry-run` | Build the prompt, write it to disk, make no API call |
| `--refresh` | Ignore caches; re-download prices and news |
| `--no-news` | Skip Alpha Vantage this run |
| `--apply` | Persist the post-trade portfolio (refused if constraints fail) |
| `--reset` | Reset `portfolio.json` to all cash |
| `--model` / `--output-mode` / `--thinking` | Override config for this run |

Exit codes: `0` success · `1` LLM call failed · `2` plan violates constraints.

---

## Disclaimer

Educational demonstration of the Anthropic API. Not investment advice. No broker
is connected and no orders are placed.

# LLM Trading Prog — an LLM as Head Portfolio Manager

A **notification-only** portfolio management agent. It downloads 15 years of
price history, reduces it to ~30 technical metrics per stock with pandas,
optionally attaches news sentiment, and asks **Claude Haiku 4.5** to act as a
Head Portfolio Manager operating under two hard constraints. It prints and
files a rebalancing plan.

> **The point of this project is to see the Anthropic API in action** — system
> prompts, forced strict tool use, structured outputs, prompt caching, and
> usage telemetry — using portfolio management as a domain that makes those
> mechanics legible. It is **not** a functional trading bot, and it is not
> investment advice.

### What it does not do

- **No broker is connected.** No order is ever placed anywhere. The only
  outputs are a console printout and files in `data/reports/`.
- **No backtest.** The agent is asked one question about one day. There is no
  walk-forward evaluation, so there is no evidence the decisions are any good.
- **No live data during reasoning.** The model sees exactly the JSON payload
  we hand it and is instructed to use nothing else.

---

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Add your key from [console.anthropic.com](https://console.anthropic.com/settings/keys)
to `.env`, then:

```bash
python3 run.py
```

To see the exact prompt without spending a token:

```bash
python3 run.py --dry-run
```

To verify the whole pipeline offline, with a mocked API response:

```bash
python3 -m tests.test_offline
```

---

## Architecture

```
                 ┌──────────────────┐
   Yahoo Finance │  src/data/       │  15y daily OHLCV, split/dividend adjusted
   (yfinance)  ──▶  prices.py       │  → data/cache/prices.parquet
                 └────────┬─────────┘
                          │  60,148 rows × 16 tickers
                 ┌────────▼─────────┐
                 │  src/features/   │  ~30 scalar metrics per ticker
                 │  technicals.py   │  + 15×15 correlation matrix
                 └────────┬─────────┘
                          │
  Alpha Vantage  ┌────────▼─────────┐
  NEWS_SENTIMENT─▶  src/agent/      │  one JSON payload:
                 │  payload.py      │  constraints + positions + metrics + news
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │  src/agent/      │  system prompt (cached, frozen)
                 │  manager.py      │  + forced strict tool use
                 │                  │  → claude-haiku-4-5
                 └────────┬─────────┘
                          │  validated plan JSON
                 ┌────────▼─────────┐
                 │  src/portfolio.py│  simulate fills, re-derive weights,
                 │                  │  VERIFY constraints in Python
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │  src/report.py   │  rich console + .json + .md
                 └──────────────────┘
```

### Files

| Path | Responsibility |
|---|---|
| `run.py` | CLI entry point; wires the seven pipeline stages together |
| `config.yaml` | Universe, data windows, constraints, LLM settings |
| `.env` | Secrets only (`ANTHROPIC_API_KEY`, `ALPHAVANTAGE_API_KEY`) — gitignored |
| `src/config.py` | Loads both of the above into a `Config` dataclass |
| `src/data/prices.py` | yfinance download, MultiIndex flattening, parquet cache |
| `src/data/news.py` | Alpha Vantage NEWS_SENTIMENT client, cache, graceful degradation |
| `src/features/technicals.py` | All pandas indicator maths |
| `src/agent/prompts.py` | The system prompt and the JSON schema for the plan |
| `src/agent/payload.py` | Builds and renders the user message |
| `src/agent/manager.py` | The Anthropic API call, both output modes, error handling |
| `src/portfolio.py` | State, mark-to-market, trade simulation, **constraint verifier** |
| `src/report.py` | Console, JSON, and Markdown output |
| `tests/test_offline.py` | 19-assertion end-to-end test with a mocked API |

---

## The Anthropic API surface being demonstrated

This is the part the project exists for. All of it lives in
`src/agent/manager.py` and `src/agent/prompts.py`.

### 1. Model

`claude-haiku-4-5` — the most recent Haiku. Configured in `config.yaml` under
`llm.model` and overridable with `--model`.

Haiku 4.5 predates two newer API features, which shapes the code:
- Extended thinking uses the older `{"type": "enabled", "budget_tokens": N}`
  form, **not** `{"type": "adaptive"}`.
- `output_config.effort` is not supported on Haiku 4.5 and is never sent.

### 2. System prompt

The user-supplied brief is the opening paragraph of `SYSTEM_PROMPT`, extended
with seven operating rules and a **metric glossary**. The glossary is load-
bearing: without it the model has to guess whether `momentum_12_1_pct` is a
price, a ratio, or a percent, and it guesses wrong often enough to matter. It
also states that `*_pct` fields are already percentages, which prevents a
100× misreading.

Rule 6 — *"Every action needs a rationale that cites at least one specific
metric value"* — is the main lever against hand-wavy output. Rule 7 pins the
model to the supplied data and forbids appeals to outside knowledge.

### 3. Forced strict tool use  *(default path)*

```python
tools=[PLAN_TOOL]                     # "strict": True
tool_choice={"type": "tool", "name": "submit_rebalance_plan"}
```

`strict: True` plus `additionalProperties: false` and a full `required` list
guarantees `tool_use.input` validates against the schema exactly — no missing
fields, no invented ones, `action` constrained to the enum
`BUY|ADD|TRIM|SELL`, `shares_delta` guaranteed to be an integer. Forcing
`tool_choice` means the model cannot answer in prose instead.

`block.input` arrives as already-parsed JSON and is read as such. It is never
string-matched — Claude 4.6+ models vary their JSON string escaping, and
matching on the serialized form is a real source of breakage.

### 4. Structured outputs  *(alternative path)*

`--output-mode json_schema` sends the same schema through
`output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}}`
instead, and the response body itself is the JSON object. Both modes are
driven from the *same* `PLAN_SCHEMA` constant, so they cannot drift apart, and
`tests/test_offline.py` asserts both extract an identical plan.

Having both is deliberate: it makes the trade-off visible. Tool use shows an
explicit `tool_use` block in the raw response and composes with other tools;
structured output is less plumbing when a single object is all you want.

### 5. Prompt caching

The system prompt is sent as a content block with
`cache_control: {"type": "ephemeral"}`.

`SYSTEM_PROMPT` is **frozen on purpose** — no dates, no portfolio values, no
run IDs. Caching is a prefix match, so a single interpolated timestamp there
would invalidate the cache on every run. Everything volatile lives in the user
message, after the breakpoint.

Honest caveat: the minimum cacheable prefix is ~1024 tokens, and the system
prompt plus tool definition sits near that line. The cache may or may not
engage. Rather than assert it works, the report prints
`cache_creation_input_tokens` and `cache_read_input_tokens` every run so you
can see for yourself.

### 6. Extended thinking

`--thinking` enables it. Two API rules are handled in code:
`temperature` is not sent when thinking is on, and `tool_choice` cannot be
forced with thinking enabled — so tool mode falls back to `auto`. When
thinking blocks come back they are captured and folded into the Markdown
report inside a `<details>` block.

### 7. Error handling

A most-specific-first chain rather than one broad catch, so retryable and
non-retryable failures stay distinguishable:
`AuthenticationError` → `NotFoundError` → `RateLimitError` → `APIStatusError`
→ `APIConnectionError`. Each maps to a `ManagerError` with an actionable
message, and `run.py` exits `1` rather than dumping a traceback.

### 8. Usage telemetry

Every run reports input/output tokens, both cache counters, `stop_reason`,
the message id, and an estimated cost at Haiku 4.5 rates ($1/$5 per MTok).
A typical run is ~5k input tokens and ~1k output, well under a cent.

---

## The pandas layer: what the model actually sees

The LLM never sees raw OHLCV. 15 tickers × 15 years is ~56,000 rows — it would
consume the context window while communicating very little. **pandas does the
arithmetic; the model does the judgement.** Each ticker is reduced to these
scalars, computed as of the last bar:

### Returns
| Metric | Definition |
|---|---|
| `ret_1w_pct`, `ret_1m_pct`, `ret_3m_pct`, `ret_6m_pct`, `ret_12m_pct` | Simple total return over 5 / 21 / 63 / 126 / 252 trading days |
| `ret_3y_ann_pct`, `ret_10y_ann_pct` | CAGR over 3 and 10 years |
| `momentum_12_1_pct` | **12-month return excluding the most recent month.** The Jegadeesh–Titman construction — the last month is dropped because short-term reversal contaminates raw 12m momentum |

### Risk
| Metric | Definition |
|---|---|
| `vol_30d_ann_pct`, `vol_90d_ann_pct`, `vol_1y_ann_pct` | Std. dev. of daily returns × √252 |
| `max_drawdown_1y_pct`, `max_drawdown_3y_pct` | Worst peak-to-trough decline in the window |
| `atr_14_pct` | Wilder ATR(14) as a percent of price — scale-free volatility |
| `beta_vs_bench`, `corr_vs_bench` | OLS beta and correlation vs SPY, 2-year daily window |
| `return_over_vol_1y` | 12m return ÷ 12m annualised vol. **Deliberately not called "Sharpe"** — no risk-free rate is subtracted, and mislabelling it would invite the model to over-read it |

### Trend and location
| Metric | Definition |
|---|---|
| `sma_50`, `sma_200`, `px_vs_sma50_pct`, `px_vs_sma200_pct` | Moving averages and distance from them |
| `golden_cross` | Boolean: 50d above 200d |
| `rsi_14` | Wilder RSI, EMA-smoothed with α = 1/14 |
| `high_52w`, `low_52w`, `pct_below_52w_high`, `pct_above_52w_low` | 52-week range position |

### Participation
| Metric | Definition |
|---|---|
| `volume_20d_vs_90d` | 20-day average volume ÷ 90-day average. Above 1.0 = unusual participation |

### Cross-sectional
A **15×15 one-year correlation matrix** is included so "diversify" is a
data-backed instruction rather than a vibe — the model can see that MSFT and
NVDA move together while XOM does not, instead of guessing from sector labels.

---

## Data sources

### Yahoo Finance (yfinance)

- **15 years** of daily bars (the brief asked for ≥10). 15y was chosen so the
  window contains both the 2020 crash and the 2022 bear market — a 10y window
  starting in 2016 would have shown the model almost nothing but a bull run.
- `auto_adjust=True` — closes are split- **and** dividend-adjusted. Raw closes
  produce fake −50% days on split dates, which would corrupt every momentum
  and volatility number downstream.
- `group_by="ticker"` returns MultiIndex columns; `_flatten()` stacks them into
  a tidy long frame so nothing downstream has to reason about MultiIndexes.
- Cached to `data/cache/prices.parquet`, re-downloaded after 12 hours.
- Coverage is checked: any ticker with fewer than 10 years of bars is reported
  by name rather than silently accepted. (META has 14.2y — it IPO'd in 2012.)
- **SPY is downloaded but never tradable.** It exists only as the beta and
  correlation benchmark, and is excluded from `constraints.tradable_tickers`.

### Alpha Vantage NEWS_SENTIMENT (optional)

A sub-agent probed the live API before this client was written. Findings, all
of which the implementation handles:

- **Errors arrive as HTTP 200** with an `{"Information": ...}` or `{"Note": ...}`
  body. `raise_for_status()` never fires on them, so the body is inspected
  explicitly.
- **`time_published` is `YYYYMMDDTHHMMSS`** (15 chars, with seconds) on the way
  out, while the `time_from` request parameter is `YYYYMMDDTHHMM` (13 chars,
  no seconds). These are different formats and mixing them fails silently.
- **`ticker_sentiment[]` scores are strings**, not numbers — `"0.140485"` — even
  though the article-level `overall_sentiment_score` is a real JSON float.
  Everything is `float()`-cast.
- **Free tier is 25 requests/day, account-wide.** `tickers=` is AND-ed, so one
  combined call would only return articles mentioning all 15 names at once.
  We therefore issue **one request per ticker — 15/day, inside the cap** — and
  cache responses for 24 hours.
- **Per-ticker sentiment is used, not article-level.** The article-level score
  covers every ticker mentioned and is much noisier; headlines are ranked by
  the ticker's own `relevance_score` and the top 5 go into the prompt.
- The `sentiment_score_definition` string spells a label `Somewhat_Bullish`
  while the data uses `Somewhat-Bullish`. The enum is never parsed out of that
  definition string.
- History depth is undocumented and could not be measured without a real key.

**Graceful degradation is total.** A missing key, an exhausted quota, or a
network failure produces a printed reason and an empty dict; the pipeline runs
on price and technicals alone and the prompt tells the model so explicitly.
Once the quota trips mid-run, remaining tickers fall back to stale cache
instead of burning further requests.

---

## Constraints: stated to the model, verified in Python

The two hard rules — **no position above 25%**, **cash never below 10%** — are
enforced in two independent places:

1. **Stated in the system prompt** and repeated in the payload's `constraints`
   block, with the schema requiring a `constraint_self_check` field.
2. **Re-derived from scratch in `src/portfolio.py`** after the model responds.
   `apply_plan()` simulates the fills at the last close (sells before buys, so
   proceeds fund purchases), `value_portfolio()` recomputes every weight, and
   `check_constraints()` returns a list of violations.

**Only step 2 is trusted.** A model asserting "constraints satisfied" is not
evidence, and the report prints the model's self-check and the Python verdict
side by side so a disagreement is visible rather than buried. If the plan
violates a rule, `run.py` exits with code `2` and `--apply` refuses to write.

This split — *model proposes, code disposes* — is the single most transferable
pattern in the project.

`apply_plan()` also enforces what the schema cannot: it refuses a buy that
exceeds available cash, refuses a sell of shares not held, and logs each
refusal as a `SKIP` line rather than failing silently.

---

## CLI reference

```bash
python3 run.py [options]
```

| Flag | Effect |
|---|---|
| `--dry-run` | Build the full prompt, write it to `data/reports/`, make **no API call** |
| `--refresh` | Ignore caches; re-download prices and news |
| `--no-news` | Skip Alpha Vantage entirely this run |
| `--apply` | Persist the simulated post-trade portfolio to `portfolio.json` (refused if the plan violates constraints) |
| `--reset` | Reset `portfolio.json` to all cash and exit |
| `--model ID` | Override `llm.model` |
| `--output-mode tool\|json_schema` | Override the structured-output mechanism |
| `--thinking` | Enable extended thinking for this run |

**Exit codes:** `0` success · `1` LLM call failed · `2` plan violates constraints.

### Portfolio state

`portfolio.json` is seeded as $100,000 all cash on first run. By default a run
**reads** it and never writes — the agent is notification-only, so it reports
what it would do without silently mutating state. `--apply` opts into
persistence, which is what you want if you're running it repeatedly and want
the model to see real evolving positions.

---

## Configuration (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `universe` | 15 mega-cap US names | Sector-spread on purpose — 7 correlated tech tickers give the model no real trade-off to make |
| `benchmark` | `SPY` | Beta/correlation reference; never tradable |
| `data.period` | `15y` | Yahoo history depth |
| `data.cache_max_age_hours` | `12` | |
| `news.articles_per_ticker` | `5` | Headlines per ticker in the prompt |
| `news.lookback_days` | `7` | |
| `news.cache_max_age_hours` | `24` | Protects the 25/day quota |
| `portfolio.starting_cash` | `100000` | |
| `portfolio.max_position_pct` | `25` | Hard constraint |
| `portfolio.min_cash_pct` | `10` | Hard constraint |
| `llm.model` | `claude-haiku-4-5` | |
| `llm.max_tokens` | `8000` | |
| `llm.temperature` | `0.2` | Dropped automatically when thinking is on |
| `llm.output_mode` | `tool` | or `json_schema` |
| `llm.prompt_caching` | `true` | |
| `llm.thinking_enabled` | `false` | |

The universe: AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, JNJ, PG, WMT,
UNH, HD, KO, CAT.

---

## Output

Three artefacts per run:

1. **Console** — a `rich` printout: market assessment, colour-coded action
   table, simulated fills, before/after weights, a green or red constraint
   verdict, risk notes, and API telemetry.
2. **`data/reports/<timestamp>.json`** — the complete run: plan, both
   valuations, execution log, violations, and full call metadata.
3. **`data/reports/<timestamp>.md`** — a human-readable report, including the
   model's self-check next to the Python verdict, and extended thinking in a
   collapsible block when enabled.

Timestamped, so history accumulates as an audit trail.

---

## Testing

```bash
python3 -m tests.test_offline
```

19 assertions covering everything except the network call itself, which is
mocked with a canned `tool_use` response. It verifies:

- the payload carries all 15 tickers, each with ≥10 years of history, plus a
  15×15 correlation matrix;
- **both** output modes extract an identical plan from the same schema;
- the request actually forces the tool, marks it `strict`, sets the cache
  breakpoint, and targets `claude-haiku-4-5`;
- fills simulate correctly and cash is debited;
- a compliant plan passes the constraint checker — **and a non-compliant one
  fails it**, catching both the position cap and the cash floor. (A checker
  that only ever passes is worthless, so the negative case is tested too.)
- the JSON report round-trips and the Markdown renders.

---

## Design decisions

**Pre-compute everything; send scalars, not series.** The dominant choice. It
keeps the prompt at ~5k tokens instead of blowing the context window, removes
arithmetic from the model's job, and makes every number auditable in pandas.

**15 sector-spread names, not the Mag-7.** Seven correlated tech tickers
produce boring, degenerate rebalancing. A spread universe forces genuine
trade-offs between momentum and diversification.

**A metric glossary in the system prompt.** Field names alone are ambiguous.
Stating that `*_pct` values are already percentages, and that
`momentum_12_1_pct` skips the last month, removes an entire class of
misreadings.

**`return_over_vol_1y`, not "Sharpe".** It isn't a Sharpe ratio — no risk-free
rate. Naming it accurately stops the model treating it as one.

**Verify constraints in Python, not in the prompt.** Both are done, but only
the Python result is trusted, and the two are printed side by side.

**Read-only by default.** A notification tool that silently mutates portfolio
state isn't notification-only. Persistence is opt-in via `--apply`.

**Frozen system prompt.** Required for prompt caching to have any chance of
engaging; all volatile content sits after the breakpoint.

**Both output mechanisms, one schema.** Tool use and structured outputs are
driven from the same `PLAN_SCHEMA` constant and tested to agree, so the
comparison is real rather than aspirational.

**News is strictly optional.** The project must run for someone who has an
Anthropic key and nothing else, so every news failure path degrades to
price-only with a printed reason.

---

## Limitations

Worth being explicit, since the failure modes are not all obvious:

- **No backtest, so no evidence of skill.** One question, one day. Nothing here
  says the model's decisions beat holding SPY.
- **Fills are simulated at the last daily close.** No slippage, no spread, no
  commission, no market impact, no intraday movement.
- **Whole shares only**, so target weights are approximate — the schema asks
  for a `target_weight_pct` but the realised weight comes from an integer
  share count.
- **Survivorship bias.** The universe is 15 companies that are mega-caps
  *today*. Their 15-year histories look excellent partly because of how they
  were selected.
- **Point-in-time metrics only.** The model sees today's values, not how they
  evolved, so it cannot see a metric deteriorating.
- **News sentiment is a vendor black box.** Alpha Vantage's scoring method is
  undisclosed; the prompt tells the model to treat it as weak evidence.
- **`avg_cost` is not tax-aware** and there is no realised-P&L ledger.
- **Single-shot.** The model gets no feedback on how previous plans performed,
  so it cannot learn from its own track record.

## Disclaimer

Educational demonstration of the Anthropic API. Not investment advice. No
broker is connected and no orders are placed. Do not trade on this output.

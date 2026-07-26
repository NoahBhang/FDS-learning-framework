# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A learning framework (Korean-language, ultralearning methodology) that builds two parallel Fraud Detection System (FDS) portfolio projects sharing one core framework:

- **`bankruptcy_fds/`** — main portfolio project. Detects asset concealment / preferential repayment / fund layering by bankruptcy debtors before filing.
- **`kaggle_bank_fds/`** — secondary project using the Kaggle PaySim dataset to detect bank transaction fraud (rule-based + ML).
- **`shared/`** — the framework both projects inherit from. This is the point of the whole repo: prove the same rule-engine abstraction generalizes across two different fraud domains.

Code comments throughout the codebase are written as extended Korean-language rationale (why a data structure or design choice was made, referencing discrete-math/algorithm background). These comments are part of the pedagogical intent of the project — preserve their style when editing nearby code rather than replacing them with terser English comments.

## Commands

Run everything from the repo root (`shared` is a local package, not pip-installed, so scripts add the root to `sys.path`).

```bash
# Install
pip install -r requirements.txt        # core (pandas, SQLAlchemy, streamlit, pytest)
pip install -r requirements-ml.txt     # adds scikit-learn, lightgbm, shap for ML phases

# Tests (bankruptcy_fds only has tests currently)
python -m pytest bankruptcy_fds/tests -v -p no:cacheprovider
python -m pytest bankruptcy_fds/tests/test_split_transfer_boundary.py -v   # single file
python -m pytest bankruptcy_fds/tests -k test_detects_exactly_five_transfers -v  # single test

# End-to-end demo (generates sample data, loads SQLite, runs both rule engines)
python scripts/run_demo.py

# SQL window-function verification
sqlite3 :memory: < database/verified_window_functions.sql

# Performance benchmark (dict vs pandas groupby, list vs set lookups)
python scripts/benchmark_fds.py

# ML phases (bankruptcy_fds), run in order from bankruptcy_fds/src/
python make_synth_case.py      # synthesize labeled case data
python detect_anomaly.py       # Phase 3: IsolationForest unsupervised scoring
python pipeline_hybrid.py      # Phase 4+5: rule+ML fusion, SHAP-based reasons, Evidence Packet CSV

# Kaggle bank FDS training (from kaggle_bank_fds/src/)
python train.py <path-to-paysim-csv> --max-step 300   # quick validation run
python train.py <path-to-paysim-csv>                  # full run

# Streamlit dashboard
streamlit run bankruptcy_fds/src/ui/streamlit_app.py
```

There is no lint/format config in the repo (no ruff/flake8/black config found) — don't invent one unless asked.

## Architecture

### The shared abstraction (`shared/`)

Everything in this repo flows through one contract:

- **`shared/rules/base_fraud_rule.py`** — `BaseFraudRule` (ABC) defines `evaluate(transactions_df, **context) -> RuleResult`. Every detection rule in both `bankruptcy_fds` and `kaggle_bank_fds` subclasses this. `RuleResult` forces `reason` (human-readable explanation) and `evidence_ids` (transaction IDs) as required fields — explainability is enforced at the type level, not bolted on. `**context` is a dict (not fixed kwargs) specifically so new rules needing different context (e.g. `filing_date`, `debtor_id`) don't force a signature change.
- **`shared/scoring/risk_scorer.py`** — `RiskScorer.aggregate()` turns a list of `RuleResult` into one `InvestigationReport` (0–100 score, grade, summary). Rules never sum scores themselves — that responsibility is deliberately separated so grading-threshold changes don't touch rule code.
- **`shared/models/`** — `Transaction` (frozen dataclass — transactions are immutable facts) and `Debtor` (mutable — investigation state accumulates: `risk_score`, `suspicious_flags`).
- **`shared/features.py`** — three-layer feature pipeline (`build_feature_matrix`) shared by both projects' ML phases: Layer A raw features (log amount, time-of-day), Layer B aggregate/rolling features (promotes rule thresholds like "5+ transfers in 30 days" into continuous features), Layer C rule-output features (existing rule engine's score as an ML input). `add_aggregate_features` sorts by `(target_account, timestamp)` before `groupby.rolling` — this ordering is load-bearing, don't remove it (see the v1.1 fix note in the file header).
- **`shared/utils/data_loader.py`** — `DataLoader` wraps a SQLAlchemy engine; all SQL lives here so table-schema changes don't leak into rule/pipeline code. Uses parameterized queries (`text(...)` with `params=`) — never string-interpolate SQL here.

### Data flow (both projects follow this identically)

```
DB (long-term memory, SQLite via schema.sql)
  → DataFrame (pandas, via DataLoader.get_transactions)
  → rules list, each rule.evaluate(df, **context) → RuleResult
  → RiskScorer.aggregate(results) → InvestigationReport
  → persisted back to detection_results table (audit trail)
```

`bankruptcy_fds/src/pipelines/fraud_detection_pipeline.py` (`FraudDetectionPipeline`) is the reference implementation of this flow: it takes an injected `DataLoader` and `rules: list` in `__init__` (dependency injection — enables swapping in fakes for tests), and contains **no detection logic itself**. Adding a new rule means adding one object to the `rules` list; the pipeline file itself should never need to change for that.

### Rule implementations (the domain-specific layer)

`bankruptcy_fds/src/rules/`:
- `split_transfer.py` — `SplitTransferRule`: threshold rule (count + sum) over a time window, using `groupby` + boolean AND.
- `related_party.py` — `RelatedPartyRule`: ratio rule (amount to family/associates ÷ total amount), not an absolute-amount rule.
- `layering.py` — `LayeringRule`: models transactions as a graph (adjacency list) and does BFS from the debtor node to detect multi-hop fund movement. BFS (not DFS) is chosen deliberately so the first path found is the shortest.

`kaggle_bank_fds/src/rules/bank_fraud_rules.py`:
- `TransferCashOutRule` — conceptually the same "trace a path" idea as `LayeringRule`, applied via a self-merge on `nameDest`/`nameOrig` instead of graph traversal (PaySim is single-hop, so a merge suffices).
- `FullBalanceTransferRule` — same "ratio, not absolute amount" idea as `RelatedPartyRule` (transfer amount ÷ prior balance ≈ 1.0).

When adding a rule to either project, subclass `BaseFraudRule`, set `rule_name`/`risk_type`/`max_score` as class attributes, and keep thresholds as `__init__` parameters rather than hardcoded constants (policy values are treated as configuration, not logic).

### Rule engine vs. ML scoring (two separate systems, fused explicitly)

`bankruptcy_fds/src/rules_mvp.py` (`RuleEngineMVP`) is a **different**, earlier pattern-scoring engine (P01–P04, continuous 0–1 `rule_score` over a feature matrix) — distinct from the `BaseFraudRule` subclasses in `rules/`. It's the "Layer C" input consumed by `shared/features.py:add_rule_features`.

`pipeline_hybrid.py` fuses rule scores and ML (`IsolationForest`) scores: `S_final = alpha * S_rule + (1 - alpha) * S_ml`, with `alpha=0.7` (rule-trusted) as the initial deployment value, meant to decrease as ML validation performance accumulates. SHAP values are converted to Korean-language reasons via `FEATURE_REASONS` to keep ML output explainable, matching the same explainability requirement the rule engine enforces via `RuleResult.reason`.

Labels (`is_suspicious` / `isFraud`) are never used as ML features — only for scoring (PR-AUC, precision@k) after the fact, since real-world debtor data has no labels. `kaggle_bank_fds/src/train.py` explicitly time-splits (`step` quantiles) rather than randomly, to avoid future information leaking into training.

### Package-name collision gotcha

`bankruptcy_fds/src` and `kaggle_bank_fds/src` are both importable as `src`, so a plain `import src...` from one will shadow the other via `sys.modules` caching. `scripts/run_demo.py` works around this with `importlib.util.spec_from_file_location` to load modules under explicit aliases (`bk_split`, `bank_rules`, etc.) — follow that pattern if writing cross-project scripts from `scripts/`. Within a single project's own code (e.g. `kaggle_bank_fds/src/train.py` importing `paysim_adapter`), plain relative-style imports plus `sys.path.append(repo_root)` for `shared` is used instead.

### Database (`database/schema.sql`)

Three tables map 1:1 to domain concepts: `debtors` (entity), `transactions` (immutable event, insert-only), `detection_results` (system judgment, kept separate from `transactions` because judgments change as rules improve while facts don't). `filing_date` lives on `debtors`, not `transactions`, to avoid update anomalies — every time-window rule computes its window from this one value. Composite index `(debtor_id, transaction_date)` backs the "this debtor, this window" query pattern that every rule uses; `sender` has its own index for `LayeringRule`'s graph traversal.

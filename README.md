# News knowledge graphs for multi-firm return forecasting

Research project studying whether knowledge graphs built from financial news:
firm-to-firm relationships (supplier, customer, competitor, acquisition,
partnership) extracted with an LLM improve return forecasting across many
firms, beyond what sector labels or return correlations already provide.

Reference: Capponi, Sidaoui & Zou, *Graph Machine Learning for Asset Pricing:
Traversing the Supply Chain* (SSRN 5031617).

## Data

- **Returns**: daily close-to-close returns for 695 NYSE-listed stocks,
  Jan 2000 – Dec 2020 (instructor-provided; balanced panel, so
  survivorship-biased and NYSE-only). Working universe: the 100 most liquid
  firms by average dollar volume, ETFs excluded; robustness runs use all
  694 (SPY held out as the market hedge).
- **News**: FNSPID (Dong et al. 2024), filtered to 517k articles about the
  panel firms; used to build and validate my own extraction pipeline.
- **Knowledge graph**: the prebuilt graph from
  [DylanSand/financial-news-kg](https://github.com/DylanSand/financial-news-kg)
  — 344k dated, typed firm-to-firm edges. Coverage of the universe is dense
  2007–2014, near-empty after; snapshots older than 6 months are treated as
  absent.

Raw files are not committed; place them in `data/raw/` (see the `src/data.py`
and `src/news.py` docstrings for the expected files).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Returns panel | `src/data.py` | Clean dates × tickers matrix, sector map, liquid universe |
| News corpus | `src/news.py` | FNSPID → one row per article with tagged tickers |
| Extraction (rules) | `src/extraction_rules.py` | Hand-written keyword + name-dictionary extractor |
| Extraction (LLM) | `src/extraction_llm.py` | LLM reads headlines → typed relations |
| Name resolution | `src/resolve.py` | Company names → tickers |
| Extraction scoring | `src/evaluate_extraction.py` | Both extractors vs 45 hand-labeled headlines |
| Graph bridge | `src/graph.py` | Snapshots → per-day adjacency matrices, no look-ahead |
| Models | `src/models.py` | AR, VAR, SparseVAR (LASSO/EN), Factor, NIRVAR, GraphVAR (hard/soft) |
| Penalty tuning | `src/tune.py` | Alpha by time-series cross-validation |
| Model validation | `src/validate_models.py` | Synthetic data with known structure |
| Backtest | `src/backtest.py` | Walk-forward loop + statistical and economic metrics |
| Main experiment | `src/graph_comp.py` | 8 models × 5 networks on one window, DM tests |
| Activity targets | `src/targets.py` | Volatility and volume panels (returns are near-unpredictable) |
| 694-firm robustness | `src/universe694_comp.py` | Same comparison over the full panel |

## Design

Every model shares one interface (`fit(window)` / `predict(last_day)`) and
runs through the same walk-forward backtest: train on the previous 4 years
only, predict the next period, slide forward. The graph in force on any day
is the most recent snapshot strictly before it. Controls are
density-matched: a sector graph built from SECTOR_PEER edges and a
correlation graph matched edge-for-edge to the news graph, so any
difference is about *which* firms are linked, not how many.

## Results (see `reports/`)

- Daily returns are near-unpredictable for every model; a simple
  autoregression capturing short-term reversal is the strongest baseline,
  and dense VAR overfits badly (OOS R² of −0.15 at 100 stocks, −2.4 at 694).
- The news graph never significantly outperforms the density-matched sector
  or correlation controls on returns, at any horizon or universe size.
- Trading volume is strongly predictable (OOS R² ≈ 0.3), but the
  predictability comes from each firm's own persistence; news links add
  nothing beyond it.
- Extraction validation: the LLM extractor achieves perfect precision and
  recall on the hand-labeled sample; the rule-based one structurally cannot
  find relations involving firms outside its dictionary.

All result tables, DM significance tests, run logs, tuning grids, and the
synthetic-data validation are in `reports/`.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."   # only for extraction_llm.py

python src/data.py               # build the returns panel
python src/news.py               # filter the news corpus
python src/validate_models.py    # verify the models on synthetic data
python src/graph_comp.py         # the main experiment

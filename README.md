# News knowledge graphs for multi-firm return forecasting

Research project studying whether knowledge graphs built from financial news:
firm-to-firm relationships (supplier, customer, competitor, acquisition,
partnership) extracted with an LLM improve return forecasting across many
firms, beyond what sector labels or return correlations already provide.

Reference: Capponi, Sidaoui & Zou, *Graph Machine Learning for Asset Pricing:
Traversing the Supply Chain* (SSRN 5031617).

## Data

Daily close-to-close returns for 695 NYSE-listed stocks, Jan 2000 – Dec 2020
(instructor-provided; every stock has data on every trading day, so the panel
is balanced but survivorship-biased). The working universe is the 100 most
liquid firms by average dollar volume, ETFs excluded.

News comes from FNSPID (Dong et al. 2024), filtered to 517,259 articles
mentioning the panel firms, Apr 2009 – Jun 2020. Graph-based models are
therefore evaluated on 2010–2020, while baselines can use the full panel.

Raw files are not committed; place them in `data/raw/` (see the `src/data.py`
and `src/news.py` docstrings for the expected files).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Returns panel | `src/data.py` | Clean dates x tickers return matrix + sector map + liquid universe |
| 2. News | `src/news.py` | Filter FNSPID to panel firms; one row per article |
| 3a. Rule extractor | `src/extraction_rules.py` | Hand-written keyword + name-dictionary rules → `edges_rules.csv` |
| 3b. LLM extractor | `src/extraction_llm.py` | Claude reads each headline → `edges_llm.csv` |
| 3c. Name resolution | `src/resolve.py` | Company names from the LLM → universe tickers |
| 3d. Extractor scoring | `src/evaluate_extraction.py` | Both extractors vs. hand labels → `reports/extraction_scores.csv` |
| 4. Baselines | `src/models.py` | AR, VAR, sparse VAR (LASSO/EN), factor model (planned) |
| 5. Graph | `src/graph.py` | Edge list -> time-varying adjacency matrices (planned) |
| 6. Graph analysis | `src/graph_analysis.py` | Degrees, centrality, stability, sector mixing (planned) |
| 7. Graph-constrained VAR | `src/models.py` | Hard mask / soft penalty from the news graph (planned) |
| 8. Evaluation | `src/backtest.py`, `src/evaluate.py` | Rolling OOS forecasts, R2, DM tests, long-short portfolios (planned) |

## Extraction results

Two extractors are built and compared: a rule-based one written from scratch,
and one that queries an LLM. Both run on the same 5,770 candidate headlines
and emit the same edge-list format, so the graphs they produce are directly
comparable.

Candidate selection matters. FNSPID's ticker tags are attached loosely: of the
24,358 articles tagged with two or more universe firms, 79% never name a
universe firm in the headline text. Requiring the name to appear in the text
cuts the pool to 5,770 and raises the hit rate from 2% to 15%.

Scored against 45 hand-labeled headlines (30 drawn at random, 15 keyword-
filtered):

| Extractor | Precision | Recall | Relation type | False edges on the 30 random |
|---|---|---|---|---|
| Rules | – | 0.00 | – | 0 |
| LLM | 1.00 | 1.00 | 7/7 | 0 |

The rule extractor finds none of the labeled relations because all seven
involve a company outside the 100-firm universe (SodaStream, Pall, Starwood,
MSD, OpenPages, Baker Hughes, Jubilant FoodWorks). A dictionary-based method
cannot name a company it has no entry for; this is a structural limit, not a
tuning failure. Both extractors produce zero false edges on the random
sample, which is itself a finding: co-mention is almost pure noise.

Resulting graphs:

| | Relations | Universe–universe edges | Unique pairs |
|---|---|---|---|
| Rules | 45 | 45 | 37 |
| LLM | 777 | 225 | 78 |

Pair-level Jaccard overlap is 0.31 — the two methods largely find different
links. Mean degree of the LLM graph is 1.56, comparable to the 0.9–1.7 that
Capponi et al. report for FactSet supply-chain data. Supplier and customer
links are rare (22 and 25 of 777): news reports acquisitions and rivalries far
more often than supply relationships, so this graph captures competitive and
partnership structure rather than a supply chain.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."      # only needed for src/extraction_llm.py

python src/data.py                  # stage 1: returns panel
python src/news.py                  # stage 2: filter news
python src/extraction_rules.py      # stage 3a
python src/extraction_llm.py        # stage 3b (costs ~$5 in API usage)
python src/evaluate_extraction.py   # stage 3d
```

Then load the panel anywhere with:

```python
from src.data import load_returns
returns = load_returns("top100")   # 5279 days x 100 tickers
```

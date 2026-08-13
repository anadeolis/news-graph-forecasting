# News knowledge graphs for multi-firm return forecasting

Research project studying whether knowledge graphs built from financial news —
firm-to-firm relationships (supplier, customer, competitor, acquisition,
partnership) extracted with an LLM — improve return forecasting across many
firms, beyond what sector labels or return correlations already provide.

Reference: Capponi, Sidaoui & Zou, *Graph Machine Learning for Asset Pricing:
Traversing the Supply Chain* (SSRN 5031617).

## Data

Daily close-to-close returns for 695 NYSE-listed stocks, Jan 2000 – Dec 2020
(instructor-provided; every stock has data on every trading day, so the panel
is balanced but survivorship-biased). The working universe is the 100 most
liquid firms by average dollar volume, ETFs excluded. Raw files are not
committed; place them in `data/raw/` (see `src/data.py` docstring for the
expected files).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Returns panel | `src/data.py` | Clean dates x tickers return matrix + sector map + liquid universe |
| 2-3. News + edge extraction | `src/extraction.py` | LLM extracts typed firm-to-firm links from news (planned) |
| 4. Baselines | `src/models.py` | AR, VAR, sparse VAR (LASSO/EN), factor model (planned) |
| 5. Graph | `src/graph.py` | Edge list -> time-varying adjacency matrices (planned) |
| 6. Graph analysis | `src/graph_analysis.py` | Degrees, centrality, stability, sector mixing (planned) |
| 7. Graph-constrained VAR | `src/models.py` | Hard mask / soft penalty from the news graph (planned) |
| 8. Evaluation | `src/backtest.py`, `src/evaluate.py` | Rolling OOS forecasts, R2, DM tests, long-short portfolios (planned) |

## Setup

```bash
pip install -r requirements.txt
python src/data.py          # builds data/processed/ from data/raw/
```

Then load the panel anywhere with:

```python
from src.data import load_returns
returns = load_returns("top100")   # 5279 days x 100 tickers
```

# Time-Series Momentum

This repository contains the full pipeline for building and running a production-grade time-series momentum strategy on U.S. equities.

📈 **Momentum Strategy**  
The strategy is based on a core concept in quantitative finance: securities that have recently outperformed (or underperformed) tend to continue doing so in the short term. We adapt this concept into a systematic, real-world trading implementation using highly liquid, optionable stocks.

## 🔧 Features

- **Point-in-Time Universe Construction**  
  - Filters historical stock data to eliminate survivorship bias.  
  - Requires at least $50M in notional volume and consistent weekly options listings.  

- **Custom Momentum Scoring**  
  - Combines risk-adjusted returns (Sharpe) and market sensitivity (Beta).  
  - Generates a momentum score to rank stocks in each monthly cycle.

- **Fully Automated Long/Short Portfolio Construction**
  - Ranks tickers monthly into deciles.
  - Rebalances monthly based on updated scores.
  - Outputs long and short baskets ready for trading or analysis.
  - Saves the current rebalance selections to `rebalance_outputs/rebalance_<YYYY-MM-DD>.csv`
    (relative to the production script) so operators can review the long/short
    baskets without rerunning the job.

## 🚀 Faster Backtests

`mtum-backtest-public.py` now supports multiple market data providers and
automatic caching so that you only download each symbol once. Key options:

- `--provider {polygon|alpaca}` to choose Polygon.io or Alpaca Markets.
- Environment variables (`POLYGON_API_KEY`, `ALPACA_API_KEY`,
  `ALPACA_SECRET_KEY`) or CLI flags to pass credentials securely.
- `--cache-dir` and `--force-refresh` to manage on-disk caching of daily bars.
- `--prefetch` with `--max-workers` to download histories in parallel when
  API limits allow.
- `--universe-csv` to load the point-in-time universe from a CSV file (useful in
  notebook environments where a database connection is unavailable).
- `--output-dir` to control where the performance chart and monthly returns
  report are saved. Use `--no-show-plot` when running in a headless
  environment.

### Momentum / volatility configuration

Both the backtest (`mtum-backtest-public.py`) and the production rebalance
script (`mtum-prod-public.py`) expose the same knobs for selecting the momentum
signal and risk model. The most common combinations are:

```bash
# Classic 12-1 momentum with the legacy beta × sharpe score
python mtum-backtest-public.py --momentum-style classic

# EWMA momentum (trend-signature style) with three half-life components
python mtum-backtest-public.py --momentum-style ewma --ewm-half-lives 63,126,252

# Sharpe-like (vol-scaled) EWMA features for smoother rankings
python mtum-backtest-public.py --momentum-style ewma_sr --ewm-half-lives 42,84,168

# EMA crossover trend filter
python mtum-backtest-public.py --momentum-style ema_crossover \
    --ema-fast-halflife 32 --ema-slow-halflife 128

# Blend EWMA Sharpe and EMA crossover (default weights are equal)
python mtum-backtest-public.py --momentum-style composite

# Switch between volatility estimators
python mtum-backtest-public.py --vol-estimator ewm --vol-ewm-halflife 20
python mtum-backtest-public.py --vol-estimator rolling --vol-window 40
python mtum-backtest-public.py --vol-estimator yang-zhang --vol-window 40

# Apply the Moreira–Muir style overlay in the performance report
python mtum-backtest-public.py --momentum-style composite \
    --apply-vol-overlay --overlay-target-vol 0.12 --overlay-halflife 63
```

Any option shown above can be supplied to the production script in exactly the
same way when you want the live rebalance output to use the new signal:

```bash
python mtum-prod-public.py --momentum-style ewma_sr --ewm-half-lives 42,84,168 \
    --vol-estimator yang-zhang --vol-window 40
```

When the scripts run they print the resolved configuration (momentum style,
volatility estimator, and overlay choice) so you can confirm the live job is
using the intended settings.

The backtest pulls the full lookback window for every symbol a single time and
reuses it across monthly iterations, dramatically reducing redundant API
requests.

## ⚙️ Production Script Configuration

`mtum-prod-public.py` now looks for credentials in the following order:

1. Existing environment variables (e.g., set via `export POLYGON_API_KEY=...`).
2. A `.env` file stored next to the script containing `KEY=value` pairs. Keys
   already present in the environment are left untouched.
3. For the universe database, a local `universe.db` SQLite file is used when
   `MTUM_DATABASE_URL` is not provided. If neither is available the script
   falls back to the bundled `historical_liquid_tickers.csv` file.

This makes it easier to run the production script from notebooks or Colab
sessions without rebuilding the environment variables each time.

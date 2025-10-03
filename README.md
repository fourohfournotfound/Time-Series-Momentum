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

## 🚀 Faster Backtests

`mtum-backtest-public.py` now supports multiple market data providers and
automatic caching so that you only download each symbol once. Key options:

- `--provider {polygon|alpaca}` to choose Polygon.io or Alpaca Markets.
- Environment variables (`POLYGON_API_KEY`, `ALPACA_API_KEY`,
  `ALPACA_SECRET_KEY`) or CLI flags to pass credentials securely.
- `--cache-dir` and `--force-refresh` to manage on-disk caching of daily bars.
- `--prefetch` with `--max-workers` to download histories in parallel when
  API limits allow.
- `--output-dir` to control where the performance chart and monthly returns
  report are saved. Use `--no-show-plot` when running in a headless
  environment.

The backtest pulls the full lookback window for every symbol a single time and
reuses it across monthly iterations, dramatically reducing redundant API
requests.

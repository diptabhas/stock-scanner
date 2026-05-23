"""
================================================================================
  BUILD CACHE  —  downloads all tickers from Price_Vol.csv + major ETFs,
                  filters to optionable only, saves to cache.parquet.

  Both breakout_from_base_alert.py and momentum_surge_alert.py read from
  this cache — so yfinance is called once per ticker, not twice.
================================================================================
"""

import os
import sys
import io
import time
import warnings
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from screener_utils import check_options, get_major_etfs

warnings.filterwarnings('ignore')


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

TICKER_FILE   = 'Price_Vol.csv'
TICKER_COLUMN = 'Symbol'
DATA_PERIOD   = '5y'        # 5y — enough for breakout_from_base (needs most history)
CACHE_FILE    = 'cache.parquet'
MAX_WORKERS   = 8

OPTIONS_EXPIRY_CUTOFF = None   # None = any options qualify
                               # e.g. '2026-06-20' to require expiry >= date

# ==============================================================================
#  END CONFIGURATION
# ==============================================================================


# ==============================================================================
#  TICKER LOADING
# ==============================================================================

def load_tickers():
    try:
        df = pd.read_csv(TICKER_FILE, dtype=str)
        df.columns = df.columns.str.strip()
        col = TICKER_COLUMN if TICKER_COLUMN in df.columns else df.columns[0]
        tickers = df[col].dropna().str.strip().str.upper().tolist()
        tickers = [t.replace('$', '').replace('.', '-') for t in tickers]
        tickers = sorted(set(tickers))
        print(f'  ✓ CSV     : {len(tickers)} tickers from {TICKER_FILE}')
        return tickers
    except FileNotFoundError:
        print(f'✗ {TICKER_FILE} not found.')
        sys.exit(1)
    except Exception as e:
        print(f'✗ Error reading {TICKER_FILE}: {e}')
        sys.exit(1)


# ==============================================================================
#  DOWNLOAD + OPTIONS CHECK (one call per ticker)
# ==============================================================================

def fetch(ticker):
    """
    Downloads OHLCV history and checks options in one function.
    Returns (ticker, df, last_expiry) or (ticker, None, None) on failure.
    """
    # ── Options check ────────────────────────────────────────────────────────
    last_expiry = check_options(ticker, OPTIONS_EXPIRY_CUTOFF)
    if last_expiry is None:
        return ticker, None, None   # not optionable — skip download entirely

    # ── OHLCV download ───────────────────────────────────────────────────────
    try:
        old_stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            raw = yf.Ticker(ticker).history(
                period=DATA_PERIOD, interval='1d',
                auto_adjust=True, timeout=15
            )
        finally:
            sys.stderr = old_stderr

        if raw is None or raw.empty:
            return ticker, None, last_expiry

        raw.columns = [c.strip() for c in raw.columns]
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in raw.columns]
        if len(cols) < 5:
            return ticker, None, last_expiry

        raw = raw[cols].apply(pd.to_numeric, errors='coerce').dropna()
        if len(raw) < 252:
            return ticker, None, last_expiry

        raw['Ticker']      = ticker
        raw['Last_Expiry'] = last_expiry
        return ticker, raw, last_expiry

    except Exception:
        return ticker, None, last_expiry


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print('\n' + '=' * 72)
    print('   BUILD CACHE')
    print('=' * 72)
    print(f'  Data period    : {DATA_PERIOD} daily')
    print(f'  Options filter : {"any expiry" if OPTIONS_EXPIRY_CUTOFF is None else f">= {OPTIONS_EXPIRY_CUTOFF}"}')
    print(f'  Output file    : {CACHE_FILE}')
    print('=' * 72)

    # ── Build ticker universe ─────────────────────────────────────────────────
    csv_tickers = load_tickers()
    etf_tickers = get_major_etfs()
    all_tickers = sorted(set(csv_tickers + etf_tickers))
    print(f'  ✓ Combined  : {len(all_tickers)} unique tickers (CSV + ETFs)\n')

    # ── Fetch all ─────────────────────────────────────────────────────────────
    total          = len(all_tickers)
    frames         = []
    not_optionable = []
    no_data        = []
    start          = time.time()

    print(f'Downloading + options check for {total} tickers...')
    print('-' * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch, t): t for t in all_tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, df, expiry = fut.result()
            if expiry is None:
                not_optionable.append(ticker)
            elif df is None:
                no_data.append(ticker)
            else:
                frames.append(df)
            if i % 100 == 0:
                elapsed = time.time() - start
                eta     = (total - i) / (i / elapsed) if elapsed > 0 else 0
                print(f'  {i}/{total} done | {len(frames)} cached | ETA {eta/60:.1f} min')

    elapsed = time.time() - start
    print(f'\nDone — {total} tickers processed in {elapsed:.1f}s')
    print(f'  ✓ Cached (optionable + data) : {len(frames)}')
    print(f'  ✗ Not optionable             : {len(not_optionable)}')
    print(f'  ✗ Optionable but no data     : {len(no_data)}')

    if not frames:
        print('✗ No data cached — aborting.')
        sys.exit(1)

    # ── Save parquet ──────────────────────────────────────────────────────────
    cache_df = pd.concat(frames)
    cache_df.to_parquet(CACHE_FILE)
    size_mb = os.path.getsize(CACHE_FILE) / 1024 / 1024
    print(f'\n✓ Cache saved to {CACHE_FILE} ({size_mb:.1f} MB, {len(frames)} tickers)')


if __name__ == '__main__':
    main()

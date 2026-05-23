"""
================================================================================
  MOMENTUM SURGE ALERT  v2.0  —  GitHub Actions / Discord Edition
================================================================================
  Catches stocks launching out of a wide base into new high territory.
  Outputs bucketed by days since surge high:

    🔥 This Week:        CRWD, FTNT, DDOG
    📈 Last Week:        AMD, PANW, ARM
    💡 3 Weeks or More:  LRCX, HPE, GFS  ← feed into EMA pullback screener

  Reads ticker data from cache.parquet (built by build_cache.py).
  Sends results to Discord via DISCORD_WEBHOOK_URL environment variable.
================================================================================
"""

import os
import sys
import time
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

CACHE_FILE          = 'cache.parquet'

PCT_FROM_52W_HIGH   = 8.0
SURGE_WINDOW        = 20
SURGE_MIN_PCT       = 18.0
SURGE_MAX_PCT       = 80.0
MIN_CONSEC_UP_DAYS  = 3
RSI_PERIOD          = 7
RSI_LOOKBACK        = 30
RSI_LOW_THRESHOLD   = 55
VOLUME_EXPANSION    = 1.4
MIN_PRICE           = 20.0
MIN_AVG_VOLUME      = 500_000

MAX_WORKERS         = 8
SEND_IF_NO_RESULTS  = True

# ==============================================================================
#  END CONFIGURATION
# ==============================================================================

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
session = requests.Session()


# ==============================================================================
#  INDICATORS
# ==============================================================================

def compute_rsi(series, period=7):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ==============================================================================
#  LOAD CACHE
# ==============================================================================

def load_cache():
    try:
        df = pd.read_parquet(CACHE_FILE)
        tickers = df['Ticker'].unique().tolist()
        print(f'✓ Loaded cache from {CACHE_FILE} ({len(tickers)} tickers)')
        return df, tickers
    except FileNotFoundError:
        print(f'✗ {CACHE_FILE} not found — run build_cache.py first.')
        sys.exit(1)
    except Exception as e:
        print(f'✗ Error reading cache: {e}')
        sys.exit(1)


# ==============================================================================
#  SCREENER
# ==============================================================================

def screen(ticker, ticker_df, stats):
    df = ticker_df.copy()

    last_close = float(df['Close'].iloc[-1])
    avg_vol_50 = float(df['Volume'].tail(50).mean())

    if last_close < MIN_PRICE:
        stats['price_filter'] += 1
        return None

    if avg_vol_50 < MIN_AVG_VOLUME:
        stats['volume_filter'] += 1
        return None

    high_52w      = float(df['High'].tail(252).max())
    pct_from_high = (high_52w - last_close) / high_52w * 100
    if pct_from_high > PCT_FROM_52W_HIGH:
        stats['not_near_high'] += 1
        return None

    window    = df.tail(SURGE_WINDOW)
    surge_low = float(window['Low'].min())
    surge_pct = (last_close - surge_low) / surge_low * 100 if surge_low > 0 else 0
    if surge_pct < SURGE_MIN_PCT:
        stats['surge_fail'] += 1
        return None

    if surge_pct > SURGE_MAX_PCT:
        stats['surge_max_fail'] += 1
        return None

    closes          = window['Close'].tolist()
    surge_high      = max(closes)
    surge_high_pos  = len(closes) - 1 - closes[::-1].index(surge_high)
    days_since_high = (SURGE_WINDOW - 1) - surge_high_pos

    last5   = df['Close'].tail(6).tolist()
    up_days = sum(1 for i in range(1, len(last5)) if last5[i] > last5[i - 1])
    if up_days < MIN_CONSEC_UP_DAYS:
        stats['consec_fail'] += 1
        return None

    rsi         = compute_rsi(df['Close'], RSI_PERIOD)
    rsi_min     = float(rsi.tail(RSI_LOOKBACK).min())
    rsi_current = float(rsi.iloc[-1])
    if rsi_min >= RSI_LOW_THRESHOLD:
        stats['rsi_fail'] += 1
        return None

    surge_vol_max = float(window['Volume'].max())
    vol_ratio     = surge_vol_max / avg_vol_50 if avg_vol_50 > 0 else 0
    if VOLUME_EXPANSION > 1.0 and vol_ratio < VOLUME_EXPANSION:
        stats['volume_fail'] += 1
        return None

    return {
        'Ticker':          ticker,
        'Price':           round(last_close, 2),
        '52W_High':        round(high_52w, 2),
        'Pct_From_52WH':   round(pct_from_high, 2),
        'Surge_Pct':       round(surge_pct, 1),
        'Days_Since_High': days_since_high,
        'Up_Days_of_5':    up_days,
        'RSI7_Now':        round(rsi_current, 1),
        'RSI7_Min_30d':    round(rsi_min, 1),
        'Vol_Spike':       round(vol_ratio, 2),
    }


# ==============================================================================
#  RUN
# ==============================================================================

def run(cache_df, tickers):
    stats = {k: 0 for k in [
        'price_filter', 'volume_filter', 'not_near_high',
        'surge_fail', 'surge_max_fail', 'consec_fail',
        'rsi_fail', 'volume_fail',
    ]}

    results = []
    total   = len(tickers)
    passed  = 0
    start   = time.time()

    print(f'\nScanning {total} tickers...')
    print('-' * 60)

    grouped = {t: cache_df[cache_df['Ticker'] == t].drop(columns='Ticker')
               for t in tickers}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(screen, t, grouped[t], stats): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                results.append(res)
                passed += 1
            if i % 100 == 0:
                elapsed = time.time() - start
                eta     = (total - i) / (i / elapsed) if elapsed > 0 else 0
                print(f'  {i}/{total} | Passed: {passed} | ETA {eta/60:.1f} min')

    elapsed = time.time() - start
    print(f'\nDone — {total} tickers in {elapsed:.1f}s | Passed: {passed}')
    print('\n--- Filter breakdown ---')
    for k, v in stats.items():
        print(f'  {k:<20}: {v}')

    if results:
        return pd.DataFrame(results).sort_values('Days_Since_High').reset_index(drop=True)
    return pd.DataFrame()


# ==============================================================================
#  DISCORD
# ==============================================================================

def bucket_label(d):
    if d <= 5:  return '1 - This Week'
    if d <= 10: return '2 - Last Week'
    return              '3 - 3 Weeks or More'


def send_discord(results_df):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    if results_df.empty:
        if SEND_IF_NO_RESULTS:
            _post(f'📊 **Momentum Surge Scan** ({now}): No surges detected.')
        return

    results_df = results_df.copy()
    results_df['Bucket'] = results_df['Days_Since_High'].apply(bucket_label)

    lines = [f'🚀 **Momentum Surge Scan** ({now}) — {len(results_df)} stocks\n']

    for label, group in results_df.groupby('Bucket'):
        tickers = ', '.join(group['Ticker'].tolist())
        emoji = '🔥' if '1 -' in label else '📈' if '2 -' in label else '💡'
        lines.append(f'{emoji} **{label}** ({len(group)}): {tickers}')

    lines.append(
        '\n_Tip: Add **3 Weeks or More** tickers to your EMA pullback screener watchlist_'
    )

    _post('\n'.join(lines))


def _post(content):
    if not DISCORD_WEBHOOK_URL:
        print(content)
        return
    try:
        r = session.post(DISCORD_WEBHOOK_URL, json={'content': content}, timeout=10)
        if r.status_code not in (200, 204):
            print(f'Discord error {r.status_code}: {r.text}')
    except Exception as e:
        print(f'Discord send failed: {e}')


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print('\n' + '=' * 72)
    print('   MOMENTUM SURGE SCREENER  v2.0')
    print('=' * 72)
    print(f'  Cache file         : {CACHE_FILE}')
    print(f'  52w high proximity : within {PCT_FROM_52W_HIGH}%')
    print(f'  Surge              : {SURGE_MIN_PCT}% – {SURGE_MAX_PCT}% in {SURGE_WINDOW} bars')
    print(f'  Up-days            : >= {MIN_CONSEC_UP_DAYS} of last 5')
    print(f'  RSI({RSI_PERIOD})              : must have been < {RSI_LOW_THRESHOLD} in last {RSI_LOOKBACK} bars')
    print(f'  Volume spike       : >= {VOLUME_EXPANSION}x 50-day avg on any bar in window')
    print(f'  Universe           : price >= ${MIN_PRICE} | avg vol >= {MIN_AVG_VOLUME:,}')
    print('=' * 72)

    cache_df, tickers = load_cache()
    results_df = run(cache_df, tickers)
    send_discord(results_df)


if __name__ == '__main__':
    main()

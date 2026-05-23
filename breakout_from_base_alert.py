"""
================================================================================
  BREAKOUT FROM BASE SCREENER  —  GitHub Actions / Discord Edition
================================================================================
  Finds stocks that:
    1. Consolidated in a tight range (the "base") — flat, compressed EMAs
    2. Recently broke out above the base high on volume
    3. Gained X% above the base high within Y days

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
# Options are pre-filtered in build_cache.py — no screener_utils import needed

warnings.filterwarnings('ignore')


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

CACHE_FILE    = 'cache.parquet'

BASE_DAYS           = 30
BASE_MAX_RANGE_PCT  = 15.0
EMA_COMPRESSION_PCT = 20.0

BREAKOUT_WINDOW     = 20
BREAKOUT_GAIN_PCT   = 15.0
VOLUME_SPIKE        = 1.2

MIN_PRICE           = 50.0
MIN_AVG_VOLUME      = 500_000

# Options filtering is handled upstream in build_cache.py

MAX_WORKERS         = 8
DEBUG_MODE          = False
SEND_IF_NO_RESULTS  = True

# ==============================================================================
#  END CONFIGURATION
# ==============================================================================

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
EMA_PERIODS = [8, 21, 55, 89, 200]


# ==============================================================================
#  INDICATORS
# ==============================================================================

def compute_emas(df):
    for p in EMA_PERIODS:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df


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

    min_bars = BASE_DAYS + BREAKOUT_WINDOW + max(EMA_PERIODS)
    if len(df) < min_bars:
        stats['insufficient_data'] += 1
        if DEBUG_MODE:
            print(f'  {ticker}: ✗ Only {len(df)} bars (need {min_bars})')
        return None

    last_close = float(df['Close'].iloc[-1])
    if last_close < MIN_PRICE:
        stats['price_filter'] += 1
        return None

    avg_vol = float(df['Volume'].tail(50).mean())
    if avg_vol < MIN_AVG_VOLUME:
        stats['volume_filter'] += 1
        return None

    df = compute_emas(df)

    base = df.iloc[-(BASE_DAYS + BREAKOUT_WINDOW):-BREAKOUT_WINDOW].copy()

    base_high = float(base['High'].max())
    base_low  = float(base['Low'].min())
    if base_low == 0:
        stats['base_fail'] += 1
        return None

    range_pct = (base_high - base_low) / base_low * 100
    if range_pct > BASE_MAX_RANGE_PCT:
        stats['base_fail'] += 1
        if DEBUG_MODE:
            print(f'  {ticker}: ✗ Base range {range_pct:.1f}% > {BASE_MAX_RANGE_PCT}%')
        return None

    avg_e8  = float(base['EMA8'].mean())
    avg_e55 = float(base['EMA55'].mean())
    if avg_e55 == 0:
        stats['base_fail'] += 1
        return None

    comp_pct = abs(avg_e8 - avg_e55) / avg_e55 * 100
    if comp_pct > EMA_COMPRESSION_PCT:
        stats['base_fail'] += 1
        if DEBUG_MODE:
            print(f'  {ticker}: ✗ EMA compression {comp_pct:.1f}% > {EMA_COMPRESSION_PCT}%')
        return None

    avg_vol_base = float(base['Volume'].mean())

    recent = df.tail(BREAKOUT_WINDOW).copy()

    breakout_idx  = None
    breakout_date = None
    for i, (idx, row) in enumerate(recent.iterrows()):
        if float(row['Close']) > base_high:
            breakout_idx  = i
            breakout_date = str(idx)[:10]
            break

    if breakout_idx is None:
        stats['no_breakout'] += 1
        if DEBUG_MODE:
            curr = float(df['Close'].iloc[-1])
            print(f'  {ticker}: ✗ No close above base_high {base_high:.2f} '
                  f'(curr {curr:.2f}, {(curr - base_high) / base_high * 100:.1f}%)')
        return None

    v0 = max(0, breakout_idx - 2)
    v1 = min(len(recent), breakout_idx + 3)
    max_vol   = float(recent.iloc[v0:v1]['Volume'].max())
    vol_ratio = max_vol / avg_vol_base if avg_vol_base > 0 else 0

    if VOLUME_SPIKE > 1.0 and vol_ratio < VOLUME_SPIKE:
        stats['volume_spike_fail'] += 1
        if DEBUG_MODE:
            print(f'  {ticker}: ✗ Vol spike {vol_ratio:.2f}× < {VOLUME_SPIKE}×')
        return None

    gain = (last_close - base_high) / base_high * 100
    if gain < BREAKOUT_GAIN_PCT:
        stats['gain_fail'] += 1
        if DEBUG_MODE:
            print(f'  {ticker}: ✗ Gain {gain:.1f}% < {BREAKOUT_GAIN_PCT}%')
        return None

    days_since = BREAKOUT_WINDOW - breakout_idx - 1

    if DEBUG_MODE:
        print(f'  {ticker}: ✓ PASS — broke out {days_since}d ago, +{gain:.1f}% above base')

    return {
        'Ticker':              ticker,
        'Current_Price':       round(last_close, 2),
        'Base_High':           round(base_high, 2),
        'Base_Range_Pct':      round(range_pct, 1),
        'EMA_Compression_Pct': round(comp_pct, 1),
        'Breakout_Date':       breakout_date,
        'Days_Since_Breakout': days_since,
        'Gain_From_Base_Pct':  round(gain, 2),
        'Vol_Spike_vs_Base':   round(vol_ratio, 2),
        'Avg_Vol_50d':         int(avg_vol),
    }


# ==============================================================================
#  RUN
# ==============================================================================

def run(cache_df, tickers):
    stats = {
        'insufficient_data':  0,
        'price_filter':       0,
        'volume_filter':      0,
        'base_fail':          0,
        'no_breakout':        0,
        'volume_spike_fail':  0,
        'gain_fail':          0,
    }

    results = []
    total   = len(tickers)
    passed  = 0
    start   = time.time()

    print(f'\nScreening {total} tickers...')
    print('-' * 60)

    grouped = {t: cache_df[cache_df['Ticker'] == t].drop(columns='Ticker')
               for t in tickers}

    workers = 1 if DEBUG_MODE else MAX_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as ex:
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
        print(f'  {k:<22}: {v}')

    if results:
        return pd.DataFrame(results).sort_values('Gain_From_Base_Pct', ascending=False)
    return pd.DataFrame()


# ==============================================================================
#  DISCORD
# ==============================================================================

def send_discord(results_df):
    if not DISCORD_WEBHOOK_URL:
        print('⚠️  DISCORD_WEBHOOK_URL not set — skipping.')
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    if results_df.empty:
        if SEND_IF_NO_RESULTS:
            requests.post(DISCORD_WEBHOOK_URL, json={'content': (
                f'📊 **Breakout From Base Screener** — {now}\n'
                f'No stocks matched all criteria this run.\n'
                f'_Base: {BASE_DAYS}d ≤{BASE_MAX_RANGE_PCT}% range | '
                f'Breakout: ≥{BREAKOUT_GAIN_PCT}% gain in {BREAKOUT_WINDOW}d_'
            )}, timeout=10)
        return

    lines = [
        f'📈 **Breakout From Base Screener** — {now}',
        f'_{len(results_df)} stock(s) passed | '
        f'Base: {BASE_DAYS}d ≤{BASE_MAX_RANGE_PCT}% range | '
        f'Breakout: ≥{BREAKOUT_GAIN_PCT}% gain in last {BREAKOUT_WINDOW}d_',
        '```',
        f"{'Ticker':<7} {'Price':>7} {'Base Hi':>8} {'Gain%':>7} {'Days':>6} {'Vol':>6} {'Date':>12}",
        '-' * 58,
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['Ticker']:<7} "
            f"{row['Current_Price']:>7.2f} "
            f"{row['Base_High']:>8.2f} "
            f"{row['Gain_From_Base_Pct']:>6.1f}% "
            f"{int(row['Days_Since_Breakout']):>6} "
            f"{row['Vol_Spike_vs_Base']:>5.1f}x "
            f"{str(row['Breakout_Date']):>12}"
        )

    lines.append('```')
    tickers_csv = ', '.join(results_df['Ticker'].tolist())
    lines.append(f'**Tickers:** `{tickers_csv}`')

    message = '\n'.join(lines)
    chunks, current = [], ''
    for line in message.split('\n'):
        if len(current) + len(line) + 1 > 1950:
            chunks.append(current)
            current = line
        else:
            current += ('\n' if current else '') + line
    if current:
        chunks.append(current)

    for chunk in chunks:
        resp = requests.post(DISCORD_WEBHOOK_URL, json={'content': chunk}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f'⚠️  Discord {resp.status_code}: {resp.text}')
        time.sleep(0.5)

    print(f'✓ Discord alert sent ({len(results_df)} result(s))')


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print('\n' + '=' * 72)
    print('   BREAKOUT FROM BASE SCREENER')
    print('=' * 72)
    print(f'  Cache file       : {CACHE_FILE}')
    print(f'  Base window      : {BASE_DAYS} bars | Max range: {BASE_MAX_RANGE_PCT}% | EMA comp: {EMA_COMPRESSION_PCT}%')
    print(f'  Breakout window  : {BREAKOUT_WINDOW} bars | Min gain: {BREAKOUT_GAIN_PCT}% | Vol spike: {VOLUME_SPIKE}×')
    print(f'  Min price/volume : ${MIN_PRICE} / {MIN_AVG_VOLUME:,}')
    print('=' * 72)

    cache_df, tickers = load_cache()
    results_df = run(cache_df, tickers)

    if results_df.empty:
        print('\nNo stocks matched all criteria.')
    else:
        print(f'\n✅  {len(results_df)} stocks passed:\n')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 160)
        print(results_df.to_string(index=False))

    send_discord(results_df)


if __name__ == '__main__':
    main()

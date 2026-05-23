"""
================================================================================
  RECOVERY SCREENER  v2.0  —  GitHub Actions / Discord Edition
================================================================================
  Catches stocks in recovery after a significant decline.

  Pattern types:
    V-shape — sharp decline, single trough, strong recovery (e.g. CIBR)
    W-shape — two lows at similar levels with a mid-peak breakout (e.g. CRWD)

  Filters:
    1. Price >= $10, avg daily volume >= 500K
    2. Declined 20–70% from prior high within last 120 bars (~6 months)
    3. Recovered 75–160% of that drop (W: cleared the mid-peak by 5%+)
    4. RSI between 50 and 90

  Results bucketed by days since trough:
    📅 1 Week     📅 2 Weeks     📅 2–4 Weeks     📅 >4 Weeks

  Reads ticker data from cache.parquet (built by build_cache.py).
  Sends results to Discord via DISCORD_WEBHOOK_URL environment variable.
================================================================================
"""

import os
import sys
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

CACHE_FILE           = 'cache.parquet'

# Decline detection
DECLINE_LOOKBACK     = 120    # bars to look back (~6 months)
MIN_DECLINE_PCT      = 20.0   # minimum peak-to-trough decline (%)
MAX_DECLINE_PCT      = 70.0   # ignore crashes deeper than this

# Recovery
RECOVERY_WINDOW      = 40     # bars after trough to check for W mid-peak
MIN_RECOVERY_PCT     = 75.0   # V-shape: must recover >= X% of the drop
MAX_RECOVERY_PCT     = 160.0  # ceiling: 100% = back to prior high, 160% = 60% above it

# W-shape
W_SECOND_LOW_TOL     = 6.0    # second low within X% of first low
W_MIDPEAK_MIN_PCT    = 5.0    # mid-peak must be >= X% above first low
W_BREAKOUT_PCT       = 5.0    # current price must clear mid-peak by X%

# Volume — informational only
VOLUME_LOOKBACK      = 20
MIN_VOL_SPIKE        = 1.0    # raise to 1.3 to make volume a hard filter

# RSI
RSI_MIN              = 50.0
RSI_MAX              = 90.0

# Universe filters
MIN_PRICE            = 10.0
MIN_AVG_VOLUME       = 500_000

SEND_IF_NO_RESULTS   = True

# ==============================================================================
#  END CONFIGURATION
# ==============================================================================

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()


# ==============================================================================
#  INDICATORS
# ==============================================================================

def compute_rsi(series, period=14):
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


# ==============================================================================
#  SCREENER LOGIC
# ==============================================================================

def find_best_setup(df):
    """
    Anchor on the deepest low in the lookback window (the trough),
    then find the highest High before it as the prior peak.
    Returns (peak_price, peak_idx, trough_price, trough_idx, decline_pct) or None.
    """
    window   = df.iloc[-DECLINE_LOOKBACK:]
    tr_idx   = window['Low'].idxmin()
    tr_pos   = window.index.get_loc(tr_idx)
    tr_price = float(window['Low'].min())

    tr_pos_in_df      = df.index.get_loc(tr_idx)
    bars_after_trough = len(df) - tr_pos_in_df - 1
    if bars_after_trough < 5:
        return None

    before_trough = window.iloc[:tr_pos]
    if before_trough.empty:
        return None

    pk_idx   = before_trough['High'].idxmax()
    pk_price = float(before_trough['High'].max())
    dec_pct  = (pk_price - tr_price) / pk_price * 100

    if not (MIN_DECLINE_PCT <= dec_pct <= MAX_DECLINE_PCT):
        return None

    return pk_price, pk_idx, tr_price, tr_idx, dec_pct


def screen(ticker, df):
    """
    Run all filters for one ticker against its pre-loaded DataFrame.
    Returns result dict or None.
    """
    if df is None or len(df) < 120:
        return None

    last_close = float(df['Close'].iloc[-1])
    avg_vol_20 = float(df['Volume'].tail(20).mean())

    if last_close < MIN_PRICE:
        return None
    if avg_vol_20 < MIN_AVG_VOLUME:
        return None

    setup = find_best_setup(df)
    if setup is None:
        return None

    peak_price, peak_idx, trough_price, trough_idx, decline_pct = setup
    drop_dollars = peak_price - trough_price

    # Slice recovery window after trough
    trough_pos   = df.index.get_loc(trough_idx)
    after_trough = df.iloc[trough_pos + 1: trough_pos + RECOVERY_WINDOW + 1]

    # ── Attempt W-shape first ──────────────────────────────────────────────
    pattern = None
    extra   = {}
    if len(after_trough) >= 10:
        midpeak_price = float(after_trough['High'].max())
        midpeak_idx   = after_trough['High'].idxmax()
        midpeak_pos   = after_trough.index.get_loc(midpeak_idx)
        midpeak_pct   = (midpeak_price - trough_price) / trough_price * 100

        after_midpeak = after_trough.iloc[midpeak_pos + 1:]
        if len(after_midpeak) >= 3 and midpeak_pct >= W_MIDPEAK_MIN_PCT:
            second_low       = float(after_midpeak['Low'].min())
            sl_vs_fl         = abs(second_low - trough_price) / trough_price * 100
            gain_above_midpk = (last_close - midpeak_price) / midpeak_price * 100

            if sl_vs_fl <= W_SECOND_LOW_TOL and gain_above_midpk >= W_BREAKOUT_PCT:
                pattern = 'W'
                extra = {
                    'Midpeak_Price':     round(midpeak_price, 2),
                    'Second_Low':        round(second_low, 2),
                    'Gain_Above_Midpk%': round(gain_above_midpk, 1),
                    'Second_Low_Delta%': round(sl_vs_fl, 1),
                }

    # ── Attempt V-shape ────────────────────────────────────────────────────
    if pattern is None:
        recovery_pct = (last_close - trough_price) / drop_dollars * 100
        if recovery_pct < MIN_RECOVERY_PCT:
            return None
        pattern = 'V'
        extra = {'Recovery_Pct_of_Drop': round(recovery_pct, 1)}

    # Recovery ceiling (applies to both V and W)
    recovery_pct_of_drop = round((last_close - trough_price) / drop_dollars * 100, 1)
    if recovery_pct_of_drop > MAX_RECOVERY_PCT:
        return None

    # Volume — informational
    baseline_vol = float(df['Volume'].iloc[-VOLUME_LOOKBACK - 5: -5].mean())
    recent_vol   = float(df['Volume'].iloc[-5:].mean())
    vol_ratio    = round(recent_vol / baseline_vol, 2) if baseline_vol > 0 else 0
    vol_ok       = vol_ratio >= MIN_VOL_SPIKE

    # RSI
    rsi         = compute_rsi(df['Close'], 14)
    rsi_current = round(float(rsi.iloc[-1]), 1)
    if not (RSI_MIN <= rsi_current <= RSI_MAX):
        return None

    pct_vs_prior_high = round((last_close - peak_price) / peak_price * 100, 1)

    result = {
        'Ticker':           ticker,
        'Pattern':          pattern,
        'Price':            round(last_close, 2),
        'Prior_High':       round(peak_price, 2),
        'Pct_vs_PriorHigh': pct_vs_prior_high,
        'Trough_Price':     round(trough_price, 2),
        'Decline_Pct':      round(decline_pct, 1),
        'Trough_Date':      str(trough_idx.date()),
        'Vol_Ratio':        vol_ratio,
        'Vol_OK':           vol_ok,
        'RSI14':            rsi_current,
    }
    result.update(extra)
    return result


# ==============================================================================
#  RUN
# ==============================================================================

def run(cache_df, tickers):
    results = []
    total   = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        df  = cache_df[cache_df['Ticker'] == ticker].drop(columns='Ticker').sort_index()
        res = screen(ticker, df)
        if res:
            results.append(res)
        if i % 200 == 0:
            print(f'  {i}/{total} scanned | {len(results)} passing so far')

    print(f'\nDone. {len(results)} stocks passed out of {total}.')

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out['Trough_Date'] = pd.to_datetime(df_out['Trough_Date'])
    return df_out


# ==============================================================================
#  DISCORD
# ==============================================================================

def bucket_label(days):
    if days <= 7:   return '1 - This Week'
    if days <= 14:  return '2 - Last Week'
    if days <= 28:  return '3 - 2–4 Weeks'
    return              '4 - >4 Weeks'

BUCKET_DISPLAY = {
    '1 - This Week': '1 Week',
    '2 - Last Week': '2 Weeks',
    '3 - 2–4 Weeks': '2–4 Weeks',
    '4 - >4 Weeks':  '>4 Weeks',
}


def send_discord(results_df):
    if not DISCORD_WEBHOOK_URL:
        print('⚠️  DISCORD_WEBHOOK_URL not set — skipping Discord notification.')
        return

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    if results_df.empty:
        if SEND_IF_NO_RESULTS:
            payload = {
                'content': (
                    f'📊 **Recovery Screener** — {now}\n'
                    f'No stocks matched all criteria this run.\n'
                    f'_Decline: {MIN_DECLINE_PCT}–{MAX_DECLINE_PCT}% | '
                    f'Recovery: {MIN_RECOVERY_PCT}–{MAX_RECOVERY_PCT}% of drop | '
                    f'RSI: {RSI_MIN}–{RSI_MAX}_'
                )
            }
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        return

    today      = pd.Timestamp.utcnow().normalize()
    df         = results_df.copy()
    df['Days'] = (today - df['Trough_Date']).dt.days
    df['Bucket'] = df['Days'].apply(bucket_label)
    df = df.sort_values(['Bucket', 'Days'])

    # Count V vs W
    v_count = (df['Pattern'] == 'V').sum()
    w_count = (df['Pattern'] == 'W').sum()

    header = (
        f'🔄 **Recovery Screener** — {now}\n'
        f'_{len(df)} stocks | V: {v_count}  W: {w_count} | '
        f'Decline: {MIN_DECLINE_PCT}–{MAX_DECLINE_PCT}% | '
        f'Recovery: {MIN_RECOVERY_PCT}–{MAX_RECOVERY_PCT}% | '
        f'RSI: {RSI_MIN}–{RSI_MAX}_\n'
    )

    bucket_lines = []
    for key in sorted(BUCKET_DISPLAY.keys()):
        group = df[df['Bucket'] == key]
        if group.empty:
            continue
        tickers_str = ', '.join(group['Ticker'].tolist())
        label = BUCKET_DISPLAY[key]
        bucket_lines.append(f'📅 **{label}:** {tickers_str}')

    # Split into Discord messages (2000-char limit)
    messages = []
    current  = header
    for line in bucket_lines:
        if len(current) + len(line) + 1 > 1950:
            messages.append(current)
            current = line + '\n'
        else:
            current += line + '\n'
    if current:
        messages.append(current)

    for msg in messages:
        resp = requests.post(DISCORD_WEBHOOK_URL, json={'content': msg}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f'  Discord error: {resp.status_code} {resp.text}')


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print('=' * 72)
    print('  RECOVERY SCREENER  v2.0')
    print('=' * 72)
    print(f'  Decline          : {MIN_DECLINE_PCT}–{MAX_DECLINE_PCT}%')
    print(f'  Recovery (V)     : >= {MIN_RECOVERY_PCT}% of drop  (ceiling: {MAX_RECOVERY_PCT}%)')
    print(f'  Recovery (W)     : mid-peak breakout >= {W_BREAKOUT_PCT}%')
    print(f'  RSI(14)          : {RSI_MIN}–{RSI_MAX}')
    print(f'  Lookback         : {DECLINE_LOOKBACK} bars')
    print(f'  Universe         : price >= ${MIN_PRICE} | avg vol >= {MIN_AVG_VOLUME:,}')
    print('=' * 72)

    cache_df, tickers = load_cache()
    results_df        = run(cache_df, tickers)
    send_discord(results_df)


if __name__ == '__main__':
    main()

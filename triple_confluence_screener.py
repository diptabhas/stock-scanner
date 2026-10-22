"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        TRIPLE CONFLUENCE SCREENER  v1.0                                      ║
║        (Python port of "Triple Confluence Navigator" Pine v6 indicator)      ║
║                                                                              ║
║  A BUY/SELL fires when ALL THREE engines agree within CONFLUENCE_WINDOW:     ║
║                                                                              ║
║    1. Cardwell RSI   : RSI(14) smoothed into fast/slow lines (Kalman, RMA    ║
║                        or LinReg) — fast crosses slow                        ║
║    2. Adaptive RSI   : custom ARSI crosses the average of its filtered       ║
║                        signal line and a Supertrend run on the ARSI itself   ║
║    3. Structure      : close breaks the last confirmed swing high / low      ║
║                                                                              ║
║  Optional filters (off by default, same as the Pine script):                 ║
║    • HTF filter  — weekly bias, uses the last COMPLETED weekly bar           ║
║    • ADX filter  — blocks Cardwell signals when ADX < MIN_ADX                ║
║                                                                              ║
║  Trade plan per signal: Entry, SL (tighter of swing + buffer or ATR cap),    ║
║  TP1/TP2/TP3 at 1R/2R/3R — plus status since the signal (OPEN / TP hit /     ║
║  SL hit) and live R-multiple.                                                ║
║                                                                              ║
║  Universe : Price_Vol.csv (Symbol column) + ETFs from screener_utils.py      ║
║  Output   : Discord (DISCORD_WEBHOOK_URL env var) + triple_confluence.csv    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# CONFIGURATION — EDIT THESE
# ============================================================================
INPUT_CSV_FILES = [
    'Price_Vol.csv',
]

# -- Universe -----------------------------------------------------------------
INCLUDE_ETFS          = True      # Add ETFs from screener_utils.get_major_etfs()
MIN_PRICE             = 5.0       # Skip tickers below this price
MIN_AVG_VOL           = 500_000   # Minimum 50-day average daily volume

# -- Options filter -----------------------------------------------------------
CHECK_OPTIONS         = True      # Only report optionable tickers
OPTIONS_EXPIRY_CUTOFF = None      # None = any chain, or 'YYYY-MM-DD'

# -- Engine 1: Cardwell RSI ---------------------------------------------------
RSI_LEN               = 14
RMA_FAST_LEN          = 9
RMA_SLOW_LEN          = 45
MA_TYPE               = 'Kalman'  # 'Kalman' | 'RMA' | 'LLAMA' (= linear regression, as in Pine)

# -- Engine 2: Adaptive RSI + Supertrend ---------------------------------------
ARSI_LEN              = 14
ARSI_SIG_LEN          = 14
ARSI_FILTER_TYPE      = 'Kalman Filter'   # 'Kalman Filter' | 'RMA' | 'LLAMA'
KALMAN_PROC_N         = 0.01
KALMAN_MEAS_N         = 0.10
LLAMA_ALPHA_MIN       = 0.05
LLAMA_ALPHA_MAX       = 0.50
ST_ATR_LEN            = 14
ST_FACTOR             = 2.0

# -- Engine 3: Structure ------------------------------------------------------
SWING_LEN             = 5

# -- Confluence ---------------------------------------------------------------
CONFLUENCE_WINDOW     = 5         # All 3 engines must fire within this many bars

# -- Trade setup --------------------------------------------------------------
ATR_LEN               = 14
ATR_MULT_SL           = 1.5       # ATR cap on stop distance
SL_STRUCT_BUF         = 0.25      # Buffer beyond swing, in ATR
RR1, RR2, RR3         = 1.0, 2.0, 3.0
ENTRY_MODE            = 'signal_close'    # 'signal_close' = close of the signal bar
                                          # 'prior_close'  = close[1], exactly like Pine

# -- Optional filters (Pine defaults: both OFF) --------------------------------
USE_HTF_FILTER        = False
HTF                   = 'W'       # 'W' = weekly, 'M' = monthly
USE_ADX_FILTER        = False
ADX_LEN               = 14
MIN_ADX               = 20.0

# -- Reporting ----------------------------------------------------------------
SIGNAL_LOOKBACK_BARS  = 3         # Report signals that fired in the last N bars
                                  #   (1 = only the most recent completed bar)
COMPLETED_BARS_ONLY   = True      # Drop today's unfinished bar when run intraday,
                                  #   so signals match TradingView's confirmed bars
DROP_STOPPED_OUT      = False     # True = hide signals whose SL was already hit

# -- Data & threading ---------------------------------------------------------
HISTORY_PERIOD        = '2y'      # 2y gives enough weekly bars for the HTF filter
MAX_WORKERS           = 10
OUTPUT_CSV            = 'triple_confluence.csv'
# ============================================================================

import os
import sys
import time
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from screener_utils import get_major_etfs, check_options

warnings.filterwarnings('ignore')

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
NAN = np.nan


# ============================================================================
# PINE-EQUIVALENT PRIMITIVES
# ============================================================================

def shift(a, n=1):
    """Pine history operator a[n] — NaN where history doesn't exist."""
    out = np.full(len(a), NAN)
    if n < len(a):
        out[n:] = a[:len(a) - n]
    return out


def rma(x, n):
    """ta.rma — seeded with an SMA of the first n valid values, then Wilder."""
    out = np.full(len(x), NAN)
    alpha = 1.0 / n
    s, acc, cnt = NAN, 0.0, 0
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = s
            continue
        if np.isnan(s):
            acc += v
            cnt += 1
            if cnt == n:
                s = acc / n
        else:
            s = alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def highest(x, n):
    return pd.Series(x).rolling(n).max().to_numpy()


def lowest(x, n):
    return pd.Series(x).rolling(n).min().to_numpy()


def rsi(close, n):
    """ta.rsi"""
    ch = close - shift(close)
    up = rma(np.where(np.isnan(ch), NAN, np.maximum(ch, 0)), n)
    dn = rma(np.where(np.isnan(ch), NAN, np.maximum(-ch, 0)), n)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r[np.isnan(up) | np.isnan(dn)] = NAN
    return r


def true_range(high, low, close):
    pc = shift(close)
    tr = np.nanmax(np.vstack([high - low, np.abs(high - pc), np.abs(low - pc)]), axis=0)
    return tr


def atr(high, low, close, n):
    """ta.atr"""
    return rma(true_range(high, low, close), n)


def adx(high, low, close, n):
    """ADX from ta.dmi(n, n)"""
    up = high - shift(high)
    dn = -(low - shift(low))
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[0] = minus_dm[0] = NAN
    tr_rma = rma(true_range(high, low, close), n)
    with np.errstate(divide='ignore', invalid='ignore'):
        plus  = 100 * rma(plus_dm, n) / tr_rma
        minus = 100 * rma(minus_dm, n) / tr_rma
        s = plus + minus
        dx = np.abs(plus - minus) / np.where(s == 0, 1, s)
    return 100 * rma(dx, n)


def linreg(x, n):
    """ta.linreg(src, n, 0) — fitted value at the most recent bar."""
    out = np.full(len(x), NAN)
    if len(x) < n:
        return out
    xs = np.arange(n, dtype=float)
    xm = xs.mean()
    denom = ((xs - xm) ** 2).sum()
    win = np.lib.stride_tricks.sliding_window_view(x, n)
    ym = win.mean(axis=1)
    slope = ((xs - xm) * (win - ym[:, None])).sum(axis=1) / denom
    out[n - 1:] = ym + slope * (n - 1 - xm)
    return out


def kalman(src, proc_n, meas_n):
    """f_kalman — 1-D Kalman filter, initialised on the first valid value."""
    out = np.full(len(src), NAN)
    est, cov = NAN, 1.0
    for i, v in enumerate(src):
        if np.isnan(est):
            est = v
        cov += proc_n
        gain = cov / (cov + meas_n)
        est = est + gain * (v - est)
        cov = (1 - gain) * cov
        out[i] = est
    return out


def llama(src, n, a_min, a_max):
    """f_llama — efficiency-ratio adaptive EMA."""
    out = np.full(len(src), NAN)
    hi, lo, back = highest(src, n), lowest(src, n), shift(src, n)
    o = NAN
    for i, v in enumerate(src):
        if np.isnan(o):
            o = v
        rng = hi[i] - lo[i]
        if np.isnan(rng) or rng == 0 or np.isnan(back[i]):
            eff = 0.0
        else:
            eff = min(abs(v - back[i]) / rng, 1.0)
        o = o + (a_min + eff * (a_max - a_min)) * (v - o)
        out[i] = o
    return out


def crossover(a, b):
    """ta.crossover(a, b)"""
    return (a > b) & (shift(a) <= shift(b))


def crossunder(a, b):
    return (a < b) & (shift(a) >= shift(b))


def arsi_calc(close, n):
    """f_arsi — Adaptive RSI on 0..100 scale."""
    u, l = highest(close, n), lowest(close, n)
    r = u - l
    d = close - shift(close)
    u1, l1 = shift(u), shift(l)
    diff = np.where(u > u1, r, np.where(l < l1, -r, d))
    num = rma(diff, n)
    den = rma(np.abs(diff), n)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where((den != 0) & ~np.isnan(den), num / den * 50 + 50, 50.0)
    return out


# ============================================================================
# HIGHER-TIMEFRAME HELPER  (request.security(... expr[1], lookahead_on))
# ============================================================================

def htf_map(df, func):
    """
    Resample daily → HTF, compute func on HTF OHLC, take the PREVIOUS completed
    HTF bar's value and map it back onto every daily bar (no lookahead).
    """
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    freq = 'W-FRI' if HTF.upper().startswith('W') else 'M'
    per = idx.to_period(freq)
    g = df.groupby(per)
    htf = pd.DataFrame({
        'Open':  g['Open'].first(),
        'High':  g['High'].max(),
        'Low':   g['Low'].min(),
        'Close': g['Close'].last(),
    })
    vals = pd.Series(func(htf), index=htf.index).shift(1)
    return vals.reindex(per).to_numpy()


# ============================================================================
# FULL INDICATOR
# ============================================================================

def triple_confluence(df):
    """Runs all three engines + confluence. Returns dict of numpy arrays."""
    o = df['Open'].to_numpy(float)
    h = df['High'].to_numpy(float)
    l = df['Low'].to_numpy(float)
    c = df['Close'].to_numpy(float)
    n = len(c)

    # ── Engine 1: Cardwell RSI ───────────────────────────────────────────────
    rsi_v = rsi(c, RSI_LEN)
    if MA_TYPE == 'Kalman':
        q, r = 0.001, 0.01
        fast = np.full(n, NAN); slow = np.full(n, NAN)
        fx, fp, sx, sp = NAN, 1.0, NAN, 1.0
        ratio = RMA_SLOW_LEN / RMA_FAST_LEN
        for i in range(n):
            v = rsi_v[i]
            fk = fp / (fp + r)
            base = v if np.isnan(fx) else fx
            fx = base + fk * (v - base)
            fp = (1 - fk) * fp + q
            sk = sp / (sp + r * ratio)
            base = v if np.isnan(sx) else sx
            sx = base + sk * (v - base)
            sp = (1 - sk) * sp + q
            fast[i], slow[i] = fx, sx
    elif MA_TYPE == 'LLAMA':
        fast, slow = linreg(rsi_v, RMA_FAST_LEN), linreg(rsi_v, RMA_SLOW_LEN)
    else:
        fast, slow = rma(rsi_v, RMA_FAST_LEN), rma(rsi_v, RMA_SLOW_LEN)

    f1, s1, f2, s2 = shift(fast), shift(slow), shift(fast, 2), shift(slow, 2)
    bull1 = (f1 > s1) & (f2 <= s2)
    bear1 = (f1 < s1) & (f2 >= s2)

    atr_v  = atr(h, l, c, ATR_LEN)
    atr_v1 = np.where(np.isnan(shift(atr_v)), atr_v, shift(atr_v))

    if USE_HTF_FILTER:
        hf = htf_map(df, lambda x: rma(rsi(x['Close'].to_numpy(float), RSI_LEN), RMA_FAST_LEN))
        hs = htf_map(df, lambda x: rma(rsi(x['Close'].to_numpy(float), RSI_LEN), RMA_SLOW_LEN))
        bull1 &= hf > hs
        bear1 &= hf < hs

    adx_v = adx(h, l, c, ADX_LEN)
    if USE_ADX_FILTER:
        chop_ok = shift(adx_v) >= MIN_ADX
        bull1 &= chop_ok
        bear1 &= chop_ok

    # ── Engine 2: Adaptive RSI + Supertrend ──────────────────────────────────
    ar = arsi_calc(c, ARSI_LEN)
    if ARSI_FILTER_TYPE == 'RMA':
        ar_sig = rma(ar, ARSI_SIG_LEN)
    elif ARSI_FILTER_TYPE == 'Kalman Filter':
        ar_sig = kalman(ar, KALMAN_PROC_N, KALMAN_MEAS_N)
    else:
        ar_sig = llama(ar, ARSI_SIG_LEN, LLAMA_ALPHA_MIN, LLAMA_ALPHA_MAX)

    ar_atr = rma(np.abs(ar - shift(ar)), ST_ATR_LEN)
    upper, lower = ar + ST_FACTOR * ar_atr, ar - ST_FACTOR * ar_atr
    st = np.full(n, NAN)
    prev_st, trend_up = NAN, True
    for i in range(n):
        if np.isnan(prev_st):
            cur, trend_up = upper[i], True
        else:
            if ar[i] > prev_st:
                trend_up = True
            elif ar[i] < prev_st:
                trend_up = False
            cur = max(lower[i], prev_st) if trend_up else min(upper[i], prev_st)
            if np.isnan(lower[i]) or np.isnan(upper[i]):
                cur = NAN
        st[i] = prev_st = cur
    merged = np.where(np.isnan(st), ar_sig, (ar_sig + st) / 2)

    bull2 = crossover(shift(ar), shift(merged))
    bear2 = crossunder(shift(ar), shift(merged))
    if USE_HTF_FILTER:
        ha = htf_map(df, lambda x: arsi_calc(x['Close'].to_numpy(float), ARSI_LEN))
        bull2 &= ha >= 50
        bear2 &= ha < 50

    # ── Engine 3: Market structure ───────────────────────────────────────────
    if USE_HTF_FILTER:
        hr = htf_map(df, lambda x: rsi(x['Close'].to_numpy(float), RSI_LEN))
        htf_ok_bull, htf_ok_bear = hr > 50, hr < 50
    else:
        htf_ok_bull = htf_ok_bear = np.ones(n, bool)

    L = SWING_LEN
    bull3 = np.zeros(n, bool); bear3 = np.zeros(n, bool)
    struct_type = np.array([''] * n, dtype=object)
    swing_low_at = np.full(n, NAN); swing_high_at = np.full(n, NAN)
    last_sh = last_sl = NAN
    last_sh_bar = last_sl_bar = proc_h = proc_l = None
    trend3 = 0
    for t in range(n):
        if t >= L:
            ch, cl = h[t - L], l[t - L]
            ok_h = ok_l = True
            for i in range(0, L):
                if h[t - i] >= ch: ok_h = False
                if l[t - i] <= cl: ok_l = False
            for i in range(L + 1, 2 * L + 1):
                if t - i < 0:
                    break
                if h[t - i] > ch: ok_h = False
                if l[t - i] < cl: ok_l = False
            if ok_h:
                last_sh, last_sh_bar = ch, t - L
            if ok_l:
                last_sl, last_sl_bar = cl, t - L

        up = (not np.isnan(last_sh)) and c[t] > last_sh and htf_ok_bull[t] \
             and (proc_h is None or last_sh_bar != proc_h)
        dn = (not np.isnan(last_sl)) and c[t] < last_sl and htf_ok_bear[t] \
             and (proc_l is None or last_sl_bar != proc_l)
        if up:
            struct_type[t] = 'CHoCH' if trend3 == -1 else 'BOS'
            proc_h, trend3 = last_sh_bar, 1
        if dn:
            struct_type[t] = 'CHoCH' if trend3 == 1 else 'BOS'
            proc_l, trend3 = last_sl_bar, -1
        bull3[t], bear3[t] = up, dn
        swing_low_at[t], swing_high_at[t] = last_sl, last_sh

    # ── Confluence ───────────────────────────────────────────────────────────
    def bars_since(sig):
        out = np.empty(n, int); v = 9999
        for i in range(n):
            v = 0 if sig[i] else v + 1
            out[i] = v
        return out

    W = CONFLUENCE_WINDOW
    bull_conf = (bars_since(bull1) <= W) & (bars_since(bull2) <= W) & (bars_since(bull3) <= W)
    bear_conf = (bars_since(bear1) <= W) & (bars_since(bear2) <= W) & (bars_since(bear3) <= W)
    long_sig  = bull_conf & ~np.r_[False, bull_conf[:-1]]
    short_sig = bear_conf & ~np.r_[False, bear_conf[:-1]]

    return {
        'close': c, 'high': h, 'low': l, 'rsi': rsi_v, 'adx': adx_v,
        'atr1': atr_v1, 'long': long_sig, 'short': short_sig,
        'swing_low': swing_low_at, 'swing_high': swing_high_at,
        'struct_type': struct_type,
    }


# ============================================================================
# TRADE PLAN + STATUS SINCE SIGNAL
# ============================================================================

def build_trade(ind, t, is_long):
    c, h, l = ind['close'], ind['high'], ind['low']
    entry = c[t - 1] if ENTRY_MODE == 'prior_close' and t > 0 else c[t]
    a1 = ind['atr1'][t]
    cap = a1 * ATR_MULT_SL
    if is_long:
        sw = ind['swing_low'][t]
        sd = NAN if np.isnan(sw) else entry - (sw - a1 * SL_STRUCT_BUF)
    else:
        sw = ind['swing_high'][t]
        sd = NAN if np.isnan(sw) else (sw + a1 * SL_STRUCT_BUF) - entry
    risk = cap if (np.isnan(sd) or sd <= 0) else min(sd, cap)
    sgn = 1 if is_long else -1
    sl  = entry - sgn * risk
    tps = [entry + sgn * risk * rr for rr in (RR1, RR2, RR3)]

    # Walk forward from the bar after the signal: SL checked first (conservative)
    best_tp, stopped = 0, False
    for j in range(t + 1, len(c)):
        if (is_long and l[j] <= sl) or (not is_long and h[j] >= sl):
            stopped = True
            break
        for k in (3, 2, 1):
            if (h[j] >= tps[k - 1]) if is_long else (l[j] <= tps[k - 1]):
                best_tp = max(best_tp, k)
                break
    if best_tp and stopped:
        status = f'TP{best_tp} then SL'
    elif stopped:
        status = 'SL HIT'
    elif best_tp:
        status = f'TP{best_tp} HIT'
    else:
        status = 'OPEN'

    live_r = sgn * (c[-1] - entry) / risk if risk > 0 else NAN
    return entry, sl, tps, risk, status, live_r


# ============================================================================
# DATA
# ============================================================================

def drop_unfinished_bar(df):
    """Remove today's in-progress daily bar if the US market hasn't closed."""
    if not COMPLETED_BARS_ONLY or df.empty:
        return df
    now_et = pd.Timestamp.now(tz='America/New_York')
    last = df.index[-1]
    last_date = (last.tz_convert('America/New_York') if last.tzinfo else last).date()
    if last_date == now_et.date() and now_et.hour < 16:
        return df.iloc[:-1]
    return df


# ============================================================================
# SINGLE TICKER
# ============================================================================

def screen_single_stock(ticker):
    if '/' in ticker or '.' in ticker:
        return []
    try:
        df = yf.Ticker(ticker).history(period=HISTORY_PERIOD, timeout=15)
        if df is None or df.empty:
            return []
        df = drop_unfinished_bar(df.dropna(subset=['Open', 'High', 'Low', 'Close']))
        if len(df) < 100:
            return []

        price = float(df['Close'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-50:].mean())
        if price < MIN_PRICE or avg_vol < MIN_AVG_VOL:
            return []

        ind = triple_confluence(df)
        n = len(df)
        hits = []
        for t in range(max(0, n - SIGNAL_LOOKBACK_BARS), n):
            for is_long, key in ((True, 'long'), (False, 'short')):
                if not ind[key][t]:
                    continue
                entry, sl, tps, risk, status, live_r = build_trade(ind, t, is_long)
                if DROP_STOPPED_OUT and 'SL' in status:
                    continue
                hits.append({
                    'Ticker':      ticker,
                    'Direction':   'LONG' if is_long else 'SHORT',
                    'Signal_Date': df.index[t].strftime('%Y-%m-%d'),
                    'Bars_Ago':    n - 1 - t,
                    'Price':       round(price, 2),
                    'Entry':       round(entry, 2),
                    'SL':          round(sl, 2),
                    'TP1':         round(tps[0], 2),
                    'TP2':         round(tps[1], 2),
                    'TP3':         round(tps[2], 2),
                    'Risk_%':      round(risk / entry * 100, 2),
                    'Live_R':      round(live_r, 2),
                    'Status':      status,
                    'Structure':   ind['struct_type'][t] or '-',
                    'RSI':         round(float(ind['rsi'][-1]), 1),
                    'ADX':         round(float(ind['adx'][-1]), 1),
                })
        if not hits:
            return []

        # Options check last — it's the slowest call, and only matters for hits
        if CHECK_OPTIONS:
            last_expiry = check_options(ticker, OPTIONS_EXPIRY_CUTOFF)
            if last_expiry is None:
                return []
        return hits

    except Exception:
        return []


# ============================================================================
# DISCORD
# ============================================================================

def send_discord(content):
    if not DISCORD_WEBHOOK_URL:
        print(f'[No webhook] Message:\n{content}')
        return
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={'content': content}, timeout=10)
        if r.status_code not in (200, 204):
            print(f'Discord error {r.status_code}: {r.text}')
    except Exception as e:
        print(f'Discord send failed: {e}')


def send_discord_chunked(lines, limit=1900):
    """Discord caps messages at 2000 chars — split on line boundaries."""
    buf = ''
    for line in lines:
        if len(buf) + len(line) + 1 > limit:
            send_discord(buf)
            time.sleep(0.5)
            buf = ''
        buf += line + '\n'
    if buf.strip():
        send_discord(buf)


def format_line(r):
    arrow = '↑' if r['Direction'] == 'LONG' else '↓'
    age = 'today' if r['Bars_Ago'] == 0 else f"{r['Bars_Ago']}d ago"
    return (f"{arrow} `{r['Ticker']}` ${r['Price']} | {age} | "
            f"Entry {r['Entry']}  SL {r['SL']}  "
            f"TP {r['TP1']} / {r['TP2']} / {r['TP3']} | "
            f"Risk {r['Risk_%']}% | {r['Live_R']:+.2f}R | {r['Status']} | {r['Structure']}")


def report(results, n_tickers):
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    if results.empty:
        send_discord(f'✅ **Triple Confluence Scan** ({now_str}): '
                     f'No signals in the last {SIGNAL_LOOKBACK_BARS} bar(s) '
                     f'across {n_tickers} tickers.')
        return

    longs  = results[results['Direction'] == 'LONG']
    shorts = results[results['Direction'] == 'SHORT']
    lines = [f'🎯 **Triple Confluence** ({now_str}) — '
             f"{longs['Ticker'].nunique()} long, {shorts['Ticker'].nunique()} short "
             f'(last {SIGNAL_LOOKBACK_BARS} bar(s), {n_tickers} tickers)']
    for label, part in (('**LONG**', longs), ('**SHORT**', shorts)):
        tickers = list(dict.fromkeys(part['Ticker']))   # dedupe, keep order
        lines.append('')
        lines.append(f"{label} ({len(tickers)}): {', '.join(tickers) if tickers else 'none'}")
    send_discord_chunked(lines)


# ============================================================================
# RUN
# ============================================================================

def load_tickers():
    frames = []
    for f in INPUT_CSV_FILES:
        try:
            d = pd.read_csv(f)
            frames.append(d)
            print(f'  ✓ {f}: {len(d)} rows')
        except Exception as e:
            print(f'  ✗ {f}: {e}')
    tickers = []
    if frames:
        d = pd.concat(frames, ignore_index=True)
        tickers = [str(t).strip().upper() for t in d['Symbol'].dropna()]
    if INCLUDE_ETFS:
        tickers += get_major_etfs()
    return list(dict.fromkeys(tickers))


def run(tickers):
    results, total, start = [], len(tickers), time.time()
    print(f'\nScreening {total:,} tickers...')
    print('-' * 70)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(screen_single_stock, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            for r in fut.result():
                results.append(r)
                print(f"  {'↑' if r['Direction'] == 'LONG' else '↓'} {r['Ticker']:<8} "
                      f"{r['Direction']:<5} {r['Signal_Date']}  Entry {r['Entry']}  "
                      f"SL {r['SL']}  {r['Status']}")
            if i % 100 == 0:
                el = time.time() - start
                eta = (total - i) / (i / el) if el > 0 else 0
                print(f'  Progress {i}/{total} | ETA {eta / 60:.1f} min | signals {len(results)}')
    print(f'\nDone in {time.time() - start:.1f}s — {len(results)} signals')

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df['_d'] = df['Direction'].map({'LONG': 0, 'SHORT': 1})
    return df.sort_values(['_d', 'Bars_Ago', 'Ticker']).drop(columns='_d').reset_index(drop=True)


def main():
    print('=' * 80)
    print(' ' * 18 + 'TRIPLE CONFLUENCE SCREENER')
    print('=' * 80)
    print(f'  Cardwell     : RSI {RSI_LEN} | {MA_TYPE} {RMA_FAST_LEN}/{RMA_SLOW_LEN}')
    print(f'  Adaptive RSI : len {ARSI_LEN} | {ARSI_FILTER_TYPE} | ST {ST_ATR_LEN} x {ST_FACTOR}')
    print(f'  Structure    : swing {SWING_LEN}')
    print(f'  Confluence   : {CONFLUENCE_WINDOW} bars | lookback {SIGNAL_LOOKBACK_BARS} bar(s)')
    print(f'  Filters      : HTF {"ON (" + HTF + ")" if USE_HTF_FILTER else "off"} | '
          f'ADX {"ON >= " + str(MIN_ADX) if USE_ADX_FILTER else "off"} | '
          f'options {"ON" if CHECK_OPTIONS else "off"}')
    print('=' * 80)

    tickers = load_tickers()
    print(f'\nUniverse: {len(tickers)} tickers')
    if not tickers:
        print('✗ No tickers loaded.')
        return

    results = run(tickers)
    if not results.empty:
        results.to_csv(OUTPUT_CSV, index=False)
        print(f'✓ Saved {OUTPUT_CSV}')
        print(results.to_string(index=False))
    report(results, len(tickers))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFatal error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        sys.exit(0)

"""
================================================================================
  BREAKOUT FROM BASE SCREENER  —  GitHub Actions / Discord Edition
================================================================================
  Finds stocks that:
    1. Consolidated in a tight range (the "base") — flat, compressed EMAs
    2. Recently broke out above the base high on volume
    3. Gained X% above the base high within Y days

  Runs headlessly (no prompts).
  Sends results to Discord via DISCORD_WEBHOOK_URL environment variable.
  Designed to be triggered by cron-job.org → GitHub Actions workflow_dispatch.
================================================================================
"""

import os
import sys
import io
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from screener_utils import check_options

warnings.filterwarnings("ignore")


# ==============================================================================
#  CONFIGURATION — Edit values here
# ==============================================================================

# ── Data Source ───────────────────────────────────────────────────────────────
TICKER_FILE   = "Price_Vol.csv"  # CSV file containing tickers
TICKER_COLUMN = "Symbol"         # Column name with the ticker symbols
DATA_PERIOD   = "5y"             # History to download
TIMEFRAME     = "1d"             # "1d" = daily | "1wk" = weekly

# ── Base Detection ────────────────────────────────────────────────────────────
BASE_DAYS           = 30         # Length of the consolidation base in bars
BASE_MAX_RANGE_PCT  = 15.0       # Max high-to-low % range during the base
EMA_COMPRESSION_PCT = 20.0       # Max spread % between EMA8 and EMA55 in base

# ── Breakout Detection ────────────────────────────────────────────────────────
BREAKOUT_WINDOW     = 20         # How many recent bars to look for the breakout
BREAKOUT_GAIN_PCT   = 15.0       # Minimum gain % above the base high
VOLUME_SPIKE        = 1.2        # Breakout bar volume must be X× the base average

# ── Universe Filters ──────────────────────────────────────────────────────────
MIN_PRICE           = 50.0       # Skip stocks below this price
MIN_AVG_VOLUME      = 500_000    # Minimum 50-day average daily volume

# ── Options Filter ────────────────────────────────────────────────────────────
CHECK_OPTIONS         = True     # Only include optionable tickers
OPTIONS_EXPIRY_CUTOFF = None     # None = any options qualify
                                 # e.g. '2026-06-20' to require expiry >= date

# ── Performance ───────────────────────────────────────────────────────────────
MAX_WORKERS         = 8          # Parallel download threads
DEBUG_MODE          = False      # True = print why each stock failed (slow)

# ── Discord ───────────────────────────────────────────────────────────────────
# Webhook URL is read from the DISCORD_WEBHOOK_URL environment variable.
# Set it as a GitHub Actions secret — never hard-code it here.
SEND_IF_NO_RESULTS  = True       # Send a Discord message even when 0 stocks pass
                                 # (useful to confirm the scanner ran successfully)

# ==============================================================================
#  END OF CONFIGURATION
# ==============================================================================


EMA_PERIODS = [8, 21, 55, 89, 200]


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_emas(df):
    for p in EMA_PERIODS:
        df[f"EMA{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


# ── Download ──────────────────────────────────────────────────────────────────

def download(ticker):
    try:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            stock = yf.Ticker(ticker)
            raw   = stock.history(period=DATA_PERIOD, interval=TIMEFRAME,
                                  auto_adjust=True, timeout=15)
        finally:
            sys.stderr = old_stderr
        if raw is None or raw.empty:
            return None
        raw.columns = [c.strip() for c in raw.columns]
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        if len(cols) < 5:
            return None
        raw = raw[cols]
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna()
        return raw if len(raw) > 0 else None
    except Exception:
        return None


# ── Screener logic ────────────────────────────────────────────────────────────

def screen(ticker, stats):
    df = download(ticker)
    if df is None:
        stats["download_fail"] += 1
        return None

    min_bars = BASE_DAYS + BREAKOUT_WINDOW + max(EMA_PERIODS)
    if len(df) < min_bars:
        stats["insufficient_data"] += 1
        if DEBUG_MODE:
            print(f"  {ticker}: ✗ Only {len(df)} bars (need {min_bars})")
        return None

    last_close = float(df["Close"].iloc[-1])
    if last_close < MIN_PRICE:
        stats["price_filter"] += 1
        return None

    avg_vol = float(df["Volume"].tail(50).mean())
    if avg_vol < MIN_AVG_VOLUME:
        stats["volume_filter"] += 1
        return None

    df = compute_emas(df)

    # Base window: sits immediately before the breakout window
    base = df.iloc[-(BASE_DAYS + BREAKOUT_WINDOW):-BREAKOUT_WINDOW].copy()

    base_high = float(base["High"].max())
    base_low  = float(base["Low"].min())
    if base_low == 0:
        stats["base_fail"] += 1
        return None

    range_pct = (base_high - base_low) / base_low * 100
    if range_pct > BASE_MAX_RANGE_PCT:
        stats["base_fail"] += 1
        if DEBUG_MODE:
            print(f"  {ticker}: ✗ Base range {range_pct:.1f}% > {BASE_MAX_RANGE_PCT}%")
        return None

    avg_e8  = float(base["EMA8"].mean())
    avg_e55 = float(base["EMA55"].mean())
    if avg_e55 == 0:
        stats["base_fail"] += 1
        return None

    comp_pct = abs(avg_e8 - avg_e55) / avg_e55 * 100
    if comp_pct > EMA_COMPRESSION_PCT:
        stats["base_fail"] += 1
        if DEBUG_MODE:
            print(f"  {ticker}: ✗ EMA compression {comp_pct:.1f}% > {EMA_COMPRESSION_PCT}%")
        return None

    avg_vol_base = float(base["Volume"].mean())

    # Breakout window: most recent bars
    recent = df.tail(BREAKOUT_WINDOW).copy()

    breakout_idx  = None
    breakout_date = None
    for i, (idx, row) in enumerate(recent.iterrows()):
        if float(row["Close"]) > base_high:
            breakout_idx  = i
            breakout_date = str(idx)[:10]
            break

    if breakout_idx is None:
        stats["no_breakout"] += 1
        if DEBUG_MODE:
            curr = float(df["Close"].iloc[-1])
            print(f"  {ticker}: ✗ No close above base_high {base_high:.2f} "
                  f"(curr {curr:.2f}, {(curr - base_high) / base_high * 100:.1f}%)")
        return None

    # Volume check
    v0 = max(0, breakout_idx - 2)
    v1 = min(len(recent), breakout_idx + 3)
    max_vol   = float(recent.iloc[v0:v1]["Volume"].max())
    vol_ratio = max_vol / avg_vol_base if avg_vol_base > 0 else 0

    if VOLUME_SPIKE > 1.0 and vol_ratio < VOLUME_SPIKE:
        stats["volume_spike_fail"] += 1
        if DEBUG_MODE:
            print(f"  {ticker}: ✗ Vol spike {vol_ratio:.2f}× < {VOLUME_SPIKE}×")
        return None

    # Gain check
    gain = (last_close - base_high) / base_high * 100
    if gain < BREAKOUT_GAIN_PCT:
        stats["gain_fail"] += 1
        if DEBUG_MODE:
            print(f"  {ticker}: ✗ Gain {gain:.1f}% < {BREAKOUT_GAIN_PCT}%")
        return None

    days_since = BREAKOUT_WINDOW - breakout_idx - 1

    if DEBUG_MODE:
        print(f"  {ticker}: ✓ PASS — broke out {days_since}d ago, +{gain:.1f}% above base")

    # Options check
    if CHECK_OPTIONS:
        last_expiry = check_options(ticker, OPTIONS_EXPIRY_CUTOFF)
        if last_expiry is None:
            return None
    else:
        last_expiry = None

    return {
        "Ticker":              ticker,
        "Current_Price":       round(last_close, 2),
        "Base_High":           round(base_high, 2),
        "Base_Range_Pct":      round(range_pct, 1),
        "EMA_Compression_Pct": round(comp_pct, 1),
        "Breakout_Date":       breakout_date,
        "Days_Since_Breakout": days_since,
        "Gain_From_Base_Pct":  round(gain, 2),
        "Vol_Spike_vs_Base":   round(vol_ratio, 2),
        "Avg_Vol_50d":         int(avg_vol),
        "Last_Expiry":         last_expiry,
    }


# ── Ticker loading ────────────────────────────────────────────────────────────

def load_tickers():
    try:
        df = pd.read_csv(TICKER_FILE, dtype=str)
        df.columns = df.columns.str.strip()
        col = TICKER_COLUMN if TICKER_COLUMN in df.columns else df.columns[0]
        tickers = df[col].dropna().str.strip().str.upper().tolist()
        tickers = sorted(set(tickers))
        print(f"✓ Loaded {len(tickers)} tickers from {TICKER_FILE} (column: '{col}')")
        return tickers
    except FileNotFoundError:
        print(f"✗ File '{TICKER_FILE}' not found.")
        return []
    except Exception as e:
        print(f"✗ Error reading '{TICKER_FILE}': {e}")
        return []


# ── Run screener ──────────────────────────────────────────────────────────────

def run(ticker_list):
    stats = {
        "download_fail":      0,
        "insufficient_data":  0,
        "price_filter":       0,
        "volume_filter":      0,
        "base_fail":          0,
        "no_breakout":        0,
        "volume_spike_fail":  0,
        "gain_fail":          0,
    }

    results = []
    total   = len(ticker_list)
    passed  = 0
    start   = time.time()

    print(f"\nScreening {total} tickers...")
    print("-" * 60)

    workers = 1 if DEBUG_MODE else MAX_WORKERS

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(screen, t, stats): t for t in ticker_list}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
                passed += 1
            if i % 100 == 0:
                elapsed = time.time() - start
                rate    = i / elapsed
                eta     = (total - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{total} | Passed: {passed} | "
                      f"ETA: {eta / 60:.1f} min")

    elapsed = time.time() - start
    print(f"\nDone! {total} tickers in {elapsed:.1f}s | Passed: {passed}")

    print("\n--- Filter breakdown ---")
    print(f"  Download failed        : {stats['download_fail']}")
    print(f"  Insufficient data      : {stats['insufficient_data']}")
    print(f"  Price < ${MIN_PRICE:.0f}           : {stats['price_filter']}")
    print(f"  Volume < {MIN_AVG_VOLUME:,}   : {stats['volume_filter']}")
    print(f"  Base too wide/noisy    : {stats['base_fail']}")
    print(f"  No breakout above base : {stats['no_breakout']}")
    print(f"  Volume spike too low   : {stats['volume_spike_fail']}")
    print(f"  Gain < {BREAKOUT_GAIN_PCT:.1f}%            : {stats['gain_fail']}")

    if results:
        df = pd.DataFrame(results).sort_values("Gain_From_Base_Pct", ascending=False)
        return df
    return pd.DataFrame()


# ── Discord ───────────────────────────────────────────────────────────────────

def send_discord(results_df):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️  DISCORD_WEBHOOK_URL not set — skipping Discord notification.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    if results_df.empty:
        if SEND_IF_NO_RESULTS:
            payload = {
                "content": (
                    f"📊 **Breakout From Base Screener** — {now}\n"
                    f"No stocks matched all criteria this run.\n"
                    f"_Base: {BASE_DAYS}d ≤{BASE_MAX_RANGE_PCT}% range | "
                    f"Breakout: ≥{BREAKOUT_GAIN_PCT}% gain in {BREAKOUT_WINDOW}d_"
                )
            }
            requests.post(webhook_url, json=payload, timeout=10)
        return

    # Build message — Discord has a 2000-char limit per message
    lines = [
        f"📈 **Breakout From Base Screener** — {now}",
        f"_{len(results_df)} stock(s) passed | "
        f"Base: {BASE_DAYS}d ≤{BASE_MAX_RANGE_PCT}% range | "
        f"Breakout: ≥{BREAKOUT_GAIN_PCT}% gain in last {BREAKOUT_WINDOW}d_",
        "```",
        f"{'Ticker':<7} {'Price':>7} {'Base Hi':>8} {'Gain%':>7} {'DaysSince':>10} {'VolSpike':>9} {'Broke Out':>12}",
        "-" * 62,
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['Ticker']:<7} "
            f"{row['Current_Price']:>7.2f} "
            f"{row['Base_High']:>8.2f} "
            f"{row['Gain_From_Base_Pct']:>6.1f}% "
            f"{int(row['Days_Since_Breakout']):>10} "
            f"{row['Vol_Spike_vs_Base']:>8.2f}x "
            f"{str(row['Breakout_Date']):>12}"
        )

    lines.append("```")

    # Comma-separated tickers for easy TradingView paste
    tickers_csv = ", ".join(results_df["Ticker"].tolist())
    lines.append(f"**Tickers:** `{tickers_csv}`")

    message = "\n".join(lines)

    # Split into chunks if over Discord's 2000-char limit
    chunks = []
    current = ""
    for line in message.split("\n"):
        if len(current) + len(line) + 1 > 1950:
            chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line
    if current:
        chunks.append(current)

    for chunk in chunks:
        resp = requests.post(webhook_url, json={"content": chunk}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"⚠️  Discord returned {resp.status_code}: {resp.text}")
        time.sleep(0.5)  # avoid Discord rate limit on multi-chunk messages

    print(f"✓ Discord alert sent ({len(results_df)} result(s))")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("   BREAKOUT FROM BASE SCREENER")
    print("=" * 72)
    print(f"  Ticker file      : {TICKER_FILE}  (column: '{TICKER_COLUMN}')")
    print(f"  Data period      : {DATA_PERIOD}  |  Timeframe: {TIMEFRAME}")
    print(f"  Base window      : {BASE_DAYS} bars  |  Max range: {BASE_MAX_RANGE_PCT}%  |  EMA comp: {EMA_COMPRESSION_PCT}%")
    print(f"  Breakout window  : {BREAKOUT_WINDOW} bars  |  Min gain: {BREAKOUT_GAIN_PCT}%  |  Vol spike: {VOLUME_SPIKE}×")
    print(f"  Min price/volume : ${MIN_PRICE}  /  {MIN_AVG_VOLUME:,}")
    print(f"  Options filter   : {'ON' if CHECK_OPTIONS else 'OFF'}")
    print(f"  Debug mode       : {'ON (single-threaded)' if DEBUG_MODE else 'OFF'}")
    print("=" * 72)

    ticker_list = load_tickers()
    if not ticker_list:
        sys.exit(1)

    results_df = run(ticker_list)

    if results_df.empty:
        print("\nNo stocks matched all criteria.")
    else:
        print(f"\n✅  {len(results_df)} stocks passed:\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", "{:.2f}".format)
        print(results_df.to_string(index=False))
        print("\n── ALL TICKERS ─────────────────────────────────────────────────────────")
        print(", ".join(results_df["Ticker"].tolist()))

    send_discord(results_df)


if __name__ == "__main__":
    main()

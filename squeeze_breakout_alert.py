"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        SQUEEZE BREAKOUT ALERT  v1.0                                          ║
║                                                                              ║
║  Paste your 1a output tickers into WATCHLIST below.                          ║
║  Script checks each ticker for a live post-squeeze breakout and sends        ║
║  a Discord alert in the format:                                              ║
║                                                                              ║
║    🚀 Squeeze Breakout: NVDA, AAPL, MSFT                                    ║
║                                                                              ║
║  Breakout = squeeze ended recently + live price broke above squeeze high     ║
║           + volume confirmed on breakout bar                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION -- EDIT THESE
# ============================================================================

# Paste your 1a squeeze screener output tickers here
WATCHLIST = """
TICKER1, TICKER2, TICKER3
"""

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

# -- Squeeze definition (must match your 1a settings) ------------------------
BB_LENGTH            = 20       # Bollinger Band length
BB_MULT              = 2.0      # Bollinger Band multiplier
KC_LENGTH            = 20       # Keltner Channel length
KC_MULT              = 1.5      # Keltner Channel multiplier
MIN_SQUEEZE_PERIODS  = 15       # Minimum bars the squeeze must have lasted

# -- Breakout detection ------------------------------------------------------
BREAKOUT_WINDOW      = 20       # Bars after squeeze ends to look for breakout
VOLUME_SPIKE         = 1.5      # Breakout bar volume >= X × avg squeeze volume
                                # Set to 1.0 to disable volume check

# -- Data --------------------------------------------------------------------
DATA_PERIOD          = "1y"     # Enough history for squeeze + breakout detection

# ============================================================================
# END CONFIGURATION
# ============================================================================

WATCHLIST = [t.strip().upper() for t in WATCHLIST.split(",") if t.strip()]
session = requests.Session()


# ============================================================================
# INDICATORS
# ============================================================================

def compute_squeeze(df):
    source  = df['Close']
    basis   = source.rolling(BB_LENGTH).mean()
    dev     = BB_MULT * source.rolling(BB_LENGTH).std(ddof=0)
    upperBB = basis + dev
    lowerBB = basis - dev

    ma      = source.rolling(KC_LENGTH).mean()
    tr      = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr     = tr.rolling(KC_LENGTH).mean()
    upperKC = ma + atr * KC_MULT
    lowerKC = ma - atr * KC_MULT

    squeeze_on = (lowerBB > lowerKC) & (upperBB < upperKC)
    return squeeze_on


def find_squeeze_breakout(df, squeeze_on):
    """
    Returns breakout info dict if a valid post-squeeze breakout is detected,
    otherwise None.
    """
    sq = squeeze_on.tolist()
    n  = len(sq)

    # Find where the most recent squeeze ended
    squeeze_end_pos = None
    for i in range(n - 1, -1, -1):
        if sq[i]:
            squeeze_end_pos = i
            break

    if squeeze_end_pos is None or squeeze_end_pos == n - 1:
        return None  # No squeeze found, or still in squeeze right now

    # Count consecutive squeeze bars leading up to end
    squeeze_start_pos = squeeze_end_pos
    for i in range(squeeze_end_pos - 1, -1, -1):
        if sq[i]:
            squeeze_start_pos = i
        else:
            break

    squeeze_bars = squeeze_end_pos - squeeze_start_pos + 1
    if squeeze_bars < MIN_SQUEEZE_PERIODS:
        return None  # Squeeze wasn't long enough

    # Squeeze high and average volume during squeeze
    squeeze_slice   = df.iloc[squeeze_start_pos : squeeze_end_pos + 1]
    squeeze_high    = float(squeeze_slice['High'].max())
    avg_vol_squeeze = float(squeeze_slice['Volume'].mean())

    # Post-squeeze window
    post_start = squeeze_end_pos + 1
    post_end   = min(n, post_start + BREAKOUT_WINDOW)

    if post_start >= n:
        return None

    post_slice = df.iloc[post_start:post_end]

    # Find first close above squeeze high
    breakout_bar_pos = None
    for i, (_, row) in enumerate(post_slice.iterrows()):
        if float(row['Close']) > squeeze_high:
            breakout_bar_pos = i
            break

    if breakout_bar_pos is None:
        return None  # No breakout yet

    # Volume check
    v0 = max(0, breakout_bar_pos - 2)
    v1 = min(len(post_slice), breakout_bar_pos + 3)
    max_vol      = float(post_slice.iloc[v0:v1]['Volume'].max())
    vol_ratio    = max_vol / avg_vol_squeeze if avg_vol_squeeze > 0 else 0

    if VOLUME_SPIKE > 1.0 and vol_ratio < VOLUME_SPIKE:
        return None  # Volume didn't confirm

    current_price      = float(df['Close'].iloc[-1])
    days_since_breakout = (n - 1) - (post_start + breakout_bar_pos)
    gain_pct            = (current_price - squeeze_high) / squeeze_high * 100

    return {
        'squeeze_bars':         squeeze_bars,
        'squeeze_high':         round(squeeze_high, 2),
        'days_since_breakout':  days_since_breakout,
        'gain_pct':             round(gain_pct, 2),
        'vol_spike':            round(vol_ratio, 2),
    }


# ============================================================================
# PER-TICKER CHECK
# ============================================================================

def check_ticker(ticker):
    """
    Returns a result dict if the ticker is breaking out of a squeeze, else None.
    """
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=DATA_PERIOD, interval="1d", auto_adjust=True, timeout=15)

        if df is None or df.empty or len(df) < 100:
            return None

        df.columns = [c.strip() for c in df.columns]

        squeeze_on = compute_squeeze(df)
        breakout   = find_squeeze_breakout(df, squeeze_on)

        if breakout is None:
            return None

        return {
            'ticker':               ticker,
            'price':                round(float(df['Close'].iloc[-1]), 2),
            'squeeze_bars':         breakout['squeeze_bars'],
            'squeeze_high':         breakout['squeeze_high'],
            'days_since_breakout':  breakout['days_since_breakout'],
            'gain_pct':             breakout['gain_pct'],
            'vol_spike':            breakout['vol_spike'],
        }

    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None


# ============================================================================
# DISCORD
# ============================================================================

def send_discord(content):
    if not DISCORD_WEBHOOK_URL:
        print(f"[No webhook] Message:\n{content}")
        return
    try:
        r = session.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
        if r.status_code not in [200, 204]:
            print(f"Discord error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Discord send failed: {e}")


# ============================================================================
# MAIN
# ============================================================================

def run_once():
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    breakouts = []

    print(f"\n[{now_str}] Checking {len(WATCHLIST)} tickers for squeeze breakout...")
    print("-" * 60)

    for ticker in WATCHLIST:
        result = check_ticker(ticker)
        if result:
            breakouts.append(result)
            print(
                f"  ✓ {ticker:<8} "
                f"Sqz:{result['squeeze_bars']}d  "
                f"Brk:{result['days_since_breakout']}d ago  "
                f"+{result['gain_pct']:.1f}%  "
                f"Vol:{result['vol_spike']:.1f}x"
            )
        else:
            print(f"    {ticker:<8} no breakout", end='\r')
        time.sleep(0.15)  # rate limiting

    print(f"\n{'='*60}")
    print(f"Done — {len(breakouts)} breakouts found out of {len(WATCHLIST)} tickers")
    print(f"{'='*60}\n")

    if breakouts:
        tickers_str = ", ".join(r['ticker'] for r in breakouts)

        # Main alert line
        lines = [f"🚀 **Squeeze Breakout** ({now_str}): {tickers_str}"]

        # Detail line per ticker
        for r in breakouts:
            lines.append(
                f"  `{r['ticker']}` ${r['price']}  |  "
                f"Squeezed {r['squeeze_bars']}d  |  "
                f"Broke out {r['days_since_breakout']}d ago  |  "
                f"+{r['gain_pct']:.1f}% above base  |  "
                f"Vol spike {r['vol_spike']:.1f}×"
            )

        send_discord("\n".join(lines))

    else:
        send_discord(
            f"✅ **Squeeze Breakout Scan** ({now_str}): "
            f"No breakouts detected in {len(WATCHLIST)} tickers."
        )


if __name__ == "__main__":
    run_once()

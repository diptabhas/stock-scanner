import os
import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

WATCHLIST = """ 
AAOI, ADI, ADM, AEHR, AGX, AMAT, ARM, ARMK, AXTI, BTSG, CAT, COHR, CPRX, CRDO, 
CTVA, CVE, DELL, DVA, EBAY, FIX, FLEX, GFS, GLNG, GLW, HPE, HWM, ICHR, INTC, 
IONQ, JBL, KEYS, KGS, LSCC, MKSI, MOD, MPC, MTSI, MU, NBIS, NVDA, NVT, OUT, 
PLAB, PWR, Q, RPRX, SDRL, SEI, SFL, SNDK, SOLS, STRL, STX, TRGP, TTMI, TWLO, VIAV, WDC 
"""
WATCHLIST = [t.strip() for t in WATCHLIST.split(",") if t.strip()]

# CRON TIP: If your env var is missing, you can hardcode your webhook URL string here directly
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

# Configuration
EMA_PERIODS = [8, 21, 55, 89, 200]
TOUCH_THRESHOLD = 0.02  # +-2%
DATA_PERIOD = "2y"      # Need 2 years of daily data to accurately calculate a stable 200 EMA

session = requests.Session()

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def send_discord_message(content: str):
    if not DISCORD_WEBHOOK_URL:
        print(f"Webhook URL missing. Logged text:\n{content}")
        return
    payload = {"content": content}
    try:
        response = session.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code not in [200, 204]:
            print(f"Discord API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def check_ema_pullbacks(ticker: str) -> list[str]:
    """
    Checks if the live price touches any of the tracked daily EMAs.
    Returns a list of matching trigger strings for this ticker.
    """
    triggers = []
    try:
        t = yf.Ticker(ticker)
        
        # 1. Fetch historical daily data (2 years required for stable 200 EMA math)
        df_daily = t.history(period=DATA_PERIOD, interval="1d", raise_errors=True)
        if df_daily.empty or len(df_daily) < max(EMA_PERIODS):
            return triggers
            
        # 2. Fetch live instantaneous price data (1-minute intervals)
        df_live = t.history(period="1d", interval="1m")
        if df_live.empty:
            return triggers
            
        current_price = float(df_live["Close"].iloc[-1])
        
        # 3. Process and test each EMA boundary
        for period in EMA_PERIODS:
            ema_series = calculate_ema(df_daily["Close"], period)
            ema_val = float(ema_series.iloc[-1])
            
            threshold = ema_val * TOUCH_THRESHOLD
            lower_bound = ema_val - threshold
            upper_bound = ema_val + threshold
            
            # Check if the live spot price sits within the boundary zone
            if lower_bound <= current_price <= upper_bound:
                percent_diff = ((current_price - ema_val) / ema_val) * 100
                triggers.append(f"EMA {period} (EMA: ${ema_val:.2f} | Diff: {percent_diff:+.1f}%)")
                
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        
    return triggers

def run_once():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    triggered_messages = []
    
    print(f"Starting multi-EMA live scan for {len(WATCHLIST)} tickers...")
    for ticker in WATCHLIST:
        active_pullbacks = check_ema_pullbacks(ticker)
        if active_pullbacks:
            # Format multiple triggers cleanly if a stock touches more than one EMA boundary zone
            ema_details = ", ".join(active_pullbacks)
            triggered_messages.append(f"📈 `{ticker}` -> {ema_details}")
        time.sleep(0.1)  # Rate limiting protection
            
    if triggered_messages:
        # Construct structural visual breakdown for Discord mapping
        result_string = "\n".join(triggered_messages)
        send_discord_message(f"🔔 **Live Multi-EMA Pullback Triggers ({now_str})**:\n{result_string}")
    else:
        send_discord_message(f"✅ **Hourly Scan Complete ({now_str})**: No tickers touching tracked EMAs.")

if __name__ == "__main__":
    run_once()

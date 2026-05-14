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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

EMA_LENGTH = 55
TOUCH_THRESHOLD = 0.02  # +-2%
DATA_PERIOD = "1mo"
DATA_INTERVAL = "1h"

# Use a session object for faster subsequent HTTP requests
session = requests.Session()

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def send_discord_message(content: str):
    if not DISCORD_WEBHOOK_URL:
        print(f"Webhook URL missing. Logged text: {content}")
        return
    payload = {"content": content}
    try:
        response = session.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code not in [200, 204]:
            print(f"Discord API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def check_touch(ticker: str) -> bool:
    try:
        # Use yf.Ticker to get a clean Single-Index DataFrame for one stock
        t = yf.Ticker(ticker)
        df = t.history(period=DATA_PERIOD, interval=DATA_INTERVAL, raise_errors=True)
        
        if df.empty or len(df) < EMA_LENGTH + 2:
            return False

        # Calculate EMA safely on a flat column structure
        df["EMA55"] = calculate_ema(df["Close"], EMA_LENGTH)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        ema_val = float(latest["EMA55"])
        close_price = float(latest["Close"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        
        prev_close = float(prev["Close"])
        prev_ema = float(prev["EMA55"])
        
        # Check boundary collision
        threshold = ema_val * TOUCH_THRESHOLD
        touched = high_price >= (ema_val - threshold) and low_price <= (ema_val + threshold)
        
        # Check standard crossovers
        cross_up = prev_close < prev_ema and close_price >= ema_val
        cross_down = prev_close > prev_ema and close_price <= ema_val
        
        if touched or cross_up or cross_down:
            return True
            
    except Exception as e:
        # Avoid silencing real bugs during development
        print(f"Error processing {ticker}: {e}")
        
    return False

def run_once():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    triggered_tickers = []
    
    print(f"Starting scan for {len(WATCHLIST)} tickers...")
    for ticker in WATCHLIST:
        if check_touch(ticker):
            triggered_tickers.append(ticker)
        time.sleep(0.1) # Polite delay to avoid Yahoo Finance rate limits
            
    if triggered_tickers:
        result_string = ", ".join(triggered_tickers)
        send_discord_message(f"🔔 **EMA 55 1-Hour Triggers ({now_str})**:\n`{result_string}`")
    else:
        send_discord_message(f"✅ **Hourly Scan Complete ({now_str})**: No triggers found.")

if __name__ == "__main__":
    run_once()

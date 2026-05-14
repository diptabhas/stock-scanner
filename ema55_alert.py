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

def check_instantaneous_touch(ticker: str) -> tuple[bool, float, float]:
    """
    Checks if the live price touches the Daily 55 EMA.
    Returns: (is_triggered, current_price, daily_ema_val)
    """
    try:
        t = yf.Ticker(ticker)
        
        # 1. Fetch historical daily data for stable EMA calculations
        df_daily = t.history(period="6mo", interval="1d", raise_errors=True)
        if df_daily.empty or len(df_daily) < EMA_LENGTH:
            return False, 0.0, 0.0
            
        df_daily["EMA55"] = calculate_ema(df_daily["Close"], EMA_LENGTH)
        daily_ema_val = float(df_daily["EMA55"].iloc[-1])
        
        # 2. Fetch live instantaneous price data (1-minute intervals)
        df_live = t.history(period="1d", interval="1m")
        if df_live.empty:
            return False, 0.0, 0.0
            
        current_price = float(df_live["Close"].iloc[-1])
        
        # 3. Calculate boundary mathematically
        threshold = daily_ema_val * TOUCH_THRESHOLD
        lower_bound = daily_ema_val - threshold
        upper_bound = daily_ema_val + threshold
        
        # Trigger if the current exact instantaneous price is inside the boundary
        if lower_bound <= current_price <= upper_bound:
            return True, current_price, daily_ema_val
            
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        
    return False, 0.0, 0.0

def run_once():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    triggered_messages = []
    
    print(f"Starting live scan for {len(WATCHLIST)} tickers...")
    for ticker in WATCHLIST:
        triggered, live_price, ema_val = check_instantaneous_touch(ticker)
        if triggered:
            triggered_messages.append(f"`{ticker}` (Price: ${live_price:.2f} | Daily EMA55: ${ema_val:.2f})")
        time.sleep(0.1)  # Rate limiting protection
            
    if triggered_messages:
        result_string = "\n".join(triggered_messages)
        send_discord_message(f"🔔 **Live Tickers Touching Daily 55 EMA ({now_str})**:\n{result_string}")
    else:
        send_discord_message(f"✅ **Hourly Scan Complete ({now_str})**: No live prices touching Daily 55 EMA.")

if __name__ == "__main__":
    run_once()

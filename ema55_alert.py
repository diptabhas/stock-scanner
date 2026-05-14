import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

WATCHLIST = "SNDK,WDC,MU,STX,TTMI,COHR,GLW,MKSI,FLEX,SEI,MTSI,MOD,CAT,AMAT,NVT,BTSG,DELL,PWR,LSCC,JBL,CVE,KGS,OUT,SDRL,GFS,HPE,SOLS,ARM,TWLO,ADM,Q,EBAY,RPRX,TRGP,MPC,SFL,DVA,ARMK,CPRX,VIAV,ICHR,KEYS,GLNG"
WATCHLIST = [t.strip() for t in WATCHLIST.split(",") if t.strip()]

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

EMA_LENGTH = 55 
TOUCH_THRESHOLD = 0.02  # +-2%
DATA_PERIOD = "1mo"     
DATA_INTERVAL = "1h"    

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def send_discord_message(content: str):
    if not DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL missing.")
        return
    payload = {"content": content}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code not in [200, 204]:
            print(f"Discord API Error: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def check_touch(ticker: str) -> bool:
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False, auto_adjust=True, group_by="ticker")
        if df.empty or len(df) < EMA_LENGTH + 5:
            return False
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels:
                df = df.xs(ticker, axis=1, level=0)
            else:
                df.columns = df.columns.get_level_values(0)
        
        df["EMA55"] = calculate_ema(df["Close"], EMA_LENGTH)
        latest = df.iloc[-1]
        ema_val = float(latest["EMA55"])
        close_price = float(latest["Close"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        
        threshold = ema_val * TOUCH_THRESHOLD
        touched = high_price >= (ema_val - threshold) and low_price <= (ema_val + threshold)
        
        prev_close = float(df["Close"].iloc[-2])
        prev_ema = float(df["EMA55"].iloc[-2])
        cross_up = prev_close < prev_ema and close_price >= ema_val
        cross_down = prev_close > prev_ema and close_price <= ema_val
        
        if touched or cross_up or cross_down:
            return True
    except Exception:
        pass 
    return False

def run_once():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    triggered_tickers = []
    for ticker in WATCHLIST:
        if check_touch(ticker):
            triggered_tickers.append(ticker)
            
    if triggered_tickers:
        result_string = ",".join(triggered_tickers)
        send_discord_message(f"🔔 **EMA 55 1-Hour Triggers ({now_str})**:\n`{result_string}`")
    else:
        send_discord_message(f"✅ **Hourly Scan Complete ({now_str})**: No triggers found.")

if __name__ == "__main__":
    run_once()



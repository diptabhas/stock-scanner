import os
import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

WATCHLIST = """ 
AAOI, ACLS, ADI, ADM, AEHR, AGX, ALAB, AMAT, AMD, ARM, ARMK, ARW, ASML, AXSM, AXTI, BTSG, CAT, CBOE, CELC, CIEN, CMPR, COHR, CRCL, CRDO, CRWD, CSCO, CTVA, CVE, DDOG, DELL, DOCN, DVA, EBAY, FFIV, FIX, FLEX, FTNT, GEV, GFS, GLW, GS, HPE, HST, HUM, HUT, HWM, ICHR, INTC, JAZZ, JBL, JEPQ, KEYS, KGS, KLIC, KRYS, LAMR, LFUS, LNTH, LRCX, LSCC, MKSI, MOD, MPC, MRVL, MS, MTRN, MTSI, MU, MYRG, NBIS, NTAP, NUE, NVDA, NVT, NWPX, ON, OUT, PANW, PLAB, PWR, Q, QCOM, QRVO, RKLB, RMBS, RPRX, SANM, SDRL, SEI, SEZL, SIMO, SITM, SLOIF, SMTC, SNDK, SOLS, SOXL, SSLLF, STLD, STRL, STX, SYNA, TQQQ, TRGP, TSEM, TTMI, TWLO, TXN, ULS, VCX, VIAV, VPG, VRSN, WDC
"""
WATCHLIST = [t.strip() for t in WATCHLIST.split(",") if t.strip()]
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

EMA_PERIODS = [8, 21, 55, 89, 200]
TOUCH_THRESHOLD = 0.02  # +-2%
DATA_PERIOD = "2y"      # Required for stable 200 EMA calculation

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

def check_ema_pullbacks(ticker: str) -> list[int]:
    """
    Checks if live price touches any daily EMAs.
    Returns a list of matching EMA period integers.
    """
    triggered_periods = []
    try:
        t = yf.Ticker(ticker)
        
        # 1. Fetch historical daily data
        df_daily = t.history(period=DATA_PERIOD, interval="1d", raise_errors=True)
        if df_daily.empty or len(df_daily) < max(EMA_PERIODS):
            return triggered_periods
            
        # 2. Fetch live instantaneous price data
        df_live = t.history(period="1d", interval="1m")
        if df_live.empty:
            return triggered_periods
            
        current_price = float(df_live["Close"].iloc[-1])
        
        # 3. Process each EMA boundary
        for period in EMA_PERIODS:
            ema_series = calculate_ema(df_daily["Close"], period)
            ema_val = float(ema_series.iloc[-1])
            
            threshold = ema_val * TOUCH_THRESHOLD
            lower_bound = ema_val - threshold
            upper_bound = ema_val + threshold
            
            if lower_bound <= current_price <= upper_bound:
                triggered_periods.append(period)
                
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        
    return triggered_periods

def run_once():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Initialize dictionary structure to hold grouped tickers
    ema_groups = {period: [] for period in EMA_PERIODS}
    has_triggers = False
    
    print(f"Starting grouped multi-EMA live scan for {len(WATCHLIST)} tickers...")
    for ticker in WATCHLIST:
        matched_periods = check_ema_pullbacks(ticker)
        for period in matched_periods:
            ema_groups[period].append(ticker)
            has_triggers = True
        time.sleep(0.1)  # Rate limiting protection
            
    if has_triggers:
        # Construct exact output format requested
        message_lines = [f"🔔 **Live Multi-EMA Pullback Triggers ({now_str})**:"]
        for period in EMA_PERIODS:
            tickers_str = ", ".join(ema_groups[period]) if ema_groups[period] else "None"
            message_lines.append(f"**EMA-{period}**: {tickers_str}")
            
        send_discord_message("\n".join(message_lines))
    else:
        send_discord_message(f"✅ **Hourly Scan Complete ({now_str})**: No tickers touching tracked EMAs.")

if __name__ == "__main__":
    run_once()

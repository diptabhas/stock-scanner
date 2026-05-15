"""
screener_utils.py
=================
Shared utilities used by all five screeners:
  • get_major_etfs()       — curated list of liquid, optionable ETFs
  • is_optionable()        — checks yfinance for live options expiry dates
  • check_options()        — returns last expiry date string or None

Import in any screener:
    from screener_utils import get_major_etfs, check_options
"""

import sys
import io
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')


# ============================================================================
# OPTIONS CHECK
# ============================================================================

def check_options(ticker, cutoff_date=None):
    """
    Check whether a ticker has listed options via yfinance.

    Parameters
    ----------
    ticker      : str   — ticker symbol
    cutoff_date : str   — optional 'YYYY-MM-DD'.  If provided, the last
                          available expiry must be >= this date.
                          Pass None to only test for any options existence.

    Returns
    -------
    str   — last expiry date 'YYYY-MM-DD' if optionable (and passes cutoff)
    None  — if not optionable or fails cutoff
    """
    try:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            stock   = yf.Ticker(ticker)
            expiries = stock.options          # tuple of date strings; empty = no options
        finally:
            sys.stderr = old_stderr

        if not expiries:
            return None

        last_expiry = max(pd.to_datetime(expiries))

        if cutoff_date is not None:
            cutoff = pd.to_datetime(cutoff_date)
            if last_expiry < cutoff:
                return None

        return last_expiry.strftime('%Y-%m-%d')

    except Exception:
        return None


# ============================================================================
# MAJOR ETF LIST
# ============================================================================

def get_major_etfs():
    """
    Returns a deduplicated, sorted list of major liquid ETFs across all
    categories.  All of these have active options chains.

    Sourced from the Minervini Squeeze screener and extended with additional
    sector and thematic ETFs.
    """
    etfs = [
        # ── Broad Market ──────────────────────────────────────────────────────
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'IEMG',

        # ── Sector SPDR ───────────────────────────────────────────────────────
        'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB',
        'XLRE', 'XLC',

        # ── Tech / Growth ─────────────────────────────────────────────────────
        'VGT', 'IYW', 'IGV', 'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKQ', 'ARKX',
        'SOXX', 'SMH', 'XSD', 'IBLC', 'DTCR',

        # ── Financials ────────────────────────────────────────────────────────
        'VFH', 'KBE', 'KRE', 'IAT',

        # ── Healthcare ────────────────────────────────────────────────────────
        'VHT', 'IBB', 'XBI', 'IHI', 'IHF',

        # ── Energy ────────────────────────────────────────────────────────────
        'VDE', 'IYE', 'XOP', 'OIH', 'USO', 'UNG',

        # ── Materials / Commodities ───────────────────────────────────────────
        'VAW', 'GLD', 'SLV', 'GDX', 'GDXJ', 'DBA', 'DBC',

        # ── International ─────────────────────────────────────────────────────
        'EWJ', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWT', 'EWY', 'EWH',
        'FXI', 'INDA', 'EEM', 'EFA',

        # ── Bond / Fixed Income ───────────────────────────────────────────────
        'TLT', 'IEF', 'SHY', 'AGG', 'BND', 'LQD', 'HYG', 'JNK', 'TIP', 'MUB',

        # ── Real Estate ───────────────────────────────────────────────────────
        'VNQ', 'IYR', 'REM', 'MORT',

        # ── Volatility / Inverse ──────────────────────────────────────────────
        'VXX', 'UVXY', 'VIXY', 'SH', 'PSQ', 'DOG', 'RWM', 'SDS', 'SQQQ', 'SPXU',

        # ── Leveraged ─────────────────────────────────────────────────────────
        'TQQQ', 'SOXL', 'UPRO', 'TNA', 'SPXL', 'UDOW', 'TECL', 'FAS', 'ERX',

        # ── Thematic / Specialty ──────────────────────────────────────────────
        'JETS', 'TAN', 'ICLN', 'LIT', 'REMX', 'URA', 'HACK', 'CIBR', 'BOTZ',
        'DRIV', 'PBW', 'QCLN', 'XHB', 'ITB', 'XRT', 'XME', 'XES',

        # ── Communication Services ────────────────────────────────────────────
        'VOX', 'FCOM',

        # ── Consumer ──────────────────────────────────────────────────────────
        'VCR', 'FDIS', 'VDC', 'FSTA',

        # ── Utilities ─────────────────────────────────────────────────────────
        'VPU', 'FUTY',

        # ── Industrials ───────────────────────────────────────────────────────
        'VIS', 'FIDU',

        # ── Small / Mid Cap ───────────────────────────────────────────────────
        'IJH', 'MDY', 'VB', 'VO',

        # ── Dividend ──────────────────────────────────────────────────────────
        'VYM', 'SCHD', 'DVY', 'HDV', 'DGRO', 'SDY', 'NOBL',

        # ── Growth / Value ────────────────────────────────────────────────────
        'VUG', 'IVW', 'VTV', 'IVE', 'RPG', 'RPV',

        # ── Clean Energy ──────────────────────────────────────────────────────
        'ACES',

        # ── Crypto / Blockchain ───────────────────────────────────────────────
        'BITO', 'BITI',

        # ── Emerging / China / Europe / Developed ─────────────────────────────
        'SCHE', 'EEMV', 'MCHI', 'KWEB', 'CQQQ', 'GXC',
        'VGK', 'EZU', 'FEZ', 'HEDJ',
        'IEFA', 'SCHF',
    ]

    result = sorted(set(etfs))
    print(f"  + ETFs: {len(result)} major/liquid ETFs added")
    return result

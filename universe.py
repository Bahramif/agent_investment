"""
universe.py
===========
The single, shared candidate universe, used by vertex_deploy.py, web_ui.py and
simulate.py. ONE file, not several copies to keep in sync.

A stock's sector is not a judgement call: it comes from the data provider
(Yahoo Finance's `sector` field), mapped by PROVIDER_SECTORS below. A stock whose
provider sector maps to none of this app's sectors is not in the universe.
Re-run `python refresh_universe.py` to re-apply this to the list.

Being listed here does not make a stock buyable. Every stock has to pass the same
test (rulebook.eligibility: a year of trading history, enough liquidity), so a
recent IPO such as SPCX stays listed and becomes eligible by itself once it has a
year of history, with no exception written for it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

SECTORS: Dict[str, str] = {
    "tech_ai":     "AI / Tech",
    "energy":      "Energy",
    "industrials": "Industrials",
    "healthcare":  "Healthcare",
    "financials":  "Financials",
}

# Yahoo Finance sector -> this app's sector. Anything else is not in the universe.
PROVIDER_SECTORS: Dict[str, str] = {
    "Technology": "tech_ai",
    "Communication Services": "tech_ai",
    "Energy": "energy",
    "Industrials": "industrials",
    "Healthcare": "healthcare",
    "Financial Services": "financials",
}

# (symbol, bucket_path, quality_score) per data_ingestion.UniverseConfig. The
# quality_score here is only the fallback for a stock with no fundamentals data.
EQUITIES: List[Tuple[str, str, float]] = [
    # tech_ai (63)
    ("AAPL", "equities.tech_ai", 0.5), ("ACN", "equities.tech_ai", 0.5),
    ("ADBE", "equities.tech_ai", 0.5), ("ADI", "equities.tech_ai", 0.5),
    ("ADP", "equities.tech_ai", 0.5), ("AKAM", "equities.tech_ai", 0.5),
    ("AMAT", "equities.tech_ai", 0.5), ("AMD", "equities.tech_ai", 0.5),
    ("APH", "equities.tech_ai", 0.5), ("APP", "equities.tech_ai", 0.5),
    ("AVGO", "equities.tech_ai", 0.5), ("BR", "equities.tech_ai", 0.5),
    ("CDNS", "equities.tech_ai", 0.5), ("CDW", "equities.tech_ai", 0.5),
    ("COHR", "equities.tech_ai", 0.5), ("CPAY", "equities.tech_ai", 0.5),
    ("CRWD", "equities.tech_ai", 0.5), ("CSCO", "equities.tech_ai", 0.5),
    ("CTSH", "equities.tech_ai", 0.5), ("DDOG", "equities.tech_ai", 0.5),
    ("DELL", "equities.tech_ai", 0.5), ("FFIV", "equities.tech_ai", 0.5),
    ("FICO", "equities.tech_ai", 0.5), ("FIS", "equities.tech_ai", 0.5),
    ("FLEX", "equities.tech_ai", 0.5), ("FTNT", "equities.tech_ai", 0.5),
    ("FTV", "equities.tech_ai", 0.5), ("GDDY", "equities.tech_ai", 0.5),
    ("GLW", "equities.tech_ai", 0.5), ("GOOGL", "equities.tech_ai", 0.5),
    ("HPE", "equities.tech_ai", 0.5), ("HPQ", "equities.tech_ai", 0.5),
    ("IBM", "equities.tech_ai", 0.5), ("INTC", "equities.tech_ai", 0.5),
    ("INTU", "equities.tech_ai", 0.5), ("IT", "equities.tech_ai", 0.5),
    ("JKHY", "equities.tech_ai", 0.5), ("KEYS", "equities.tech_ai", 0.5),
    ("KLAC", "equities.tech_ai", 0.5), ("LDOS", "equities.tech_ai", 0.5),
    ("LITE", "equities.tech_ai", 0.5), ("LRCX", "equities.tech_ai", 0.5),
    ("META", "equities.tech_ai", 0.5), ("MRVL", "equities.tech_ai", 0.5),
    ("MSFT", "equities.tech_ai", 0.5), ("NFLX", "equities.tech_ai", 0.5),
    ("NTAP", "equities.tech_ai", 0.5), ("NVDA", "equities.tech_ai", 0.5),
    ("ORCL", "equities.tech_ai", 0.5), ("PAYC", "equities.tech_ai", 0.5),
    ("PAYX", "equities.tech_ai", 0.5), ("QCOM", "equities.tech_ai", 0.5),
    ("SMCI", "equities.tech_ai", 0.5), ("SNOW", "equities.tech_ai", 0.5),
    ("SNPS", "equities.tech_ai", 0.5), ("STX", "equities.tech_ai", 0.5),
    ("SWKS", "equities.tech_ai", 0.5), ("TMUS", "equities.tech_ai", 0.5),
    ("TYL", "equities.tech_ai", 0.5), ("VRSN", "equities.tech_ai", 0.5),
    ("WDAY", "equities.tech_ai", 0.5), ("WDC", "equities.tech_ai", 0.5),
    ("XYZ", "equities.tech_ai", 0.5),
    # energy (15)
    ("APA", "equities.energy", 0.5), ("BKR", "equities.energy", 0.5),
    ("COP", "equities.energy", 0.5), ("CVX", "equities.energy", 0.5),
    ("DVN", "equities.energy", 0.5), ("EOG", "equities.energy", 0.5),
    ("EQT", "equities.energy", 0.5), ("EXE", "equities.energy", 0.5),
    ("HAL", "equities.energy", 0.5), ("KMI", "equities.energy", 0.5),
    ("MPC", "equities.energy", 0.5), ("OKE", "equities.energy", 0.5),
    ("SLB", "equities.energy", 0.5), ("TRGP", "equities.energy", 0.5),
    ("XOM", "equities.energy", 0.5),
    # industrials (51)
    ("AME", "equities.industrials", 0.5), ("AOS", "equities.industrials", 0.5),
    ("AXON", "equities.industrials", 0.5), ("BA", "equities.industrials", 0.5),
    ("BLDR", "equities.industrials", 0.5), ("CARR", "equities.industrials", 0.5),
    ("CAT", "equities.industrials", 0.5), ("CHRW", "equities.industrials", 0.5),
    ("CMI", "equities.industrials", 0.5), ("CPRT", "equities.industrials", 0.5),
    ("CSX", "equities.industrials", 0.5), ("DE", "equities.industrials", 0.5),
    ("DOV", "equities.industrials", 0.5), ("EFX", "equities.industrials", 0.5),
    ("EME", "equities.industrials", 0.5), ("EMR", "equities.industrials", 0.5),
    ("ETN", "equities.industrials", 0.5), ("EXPD", "equities.industrials", 0.5),
    ("FAST", "equities.industrials", 0.5), ("FDX", "equities.industrials", 0.5),
    ("FERG", "equities.industrials", 0.5), ("GD", "equities.industrials", 0.5),
    ("GE", "equities.industrials", 0.5), ("GNRC", "equities.industrials", 0.5),
    ("HII", "equities.industrials", 0.5), ("HON", "equities.industrials", 0.5),
    ("HWM", "equities.industrials", 0.5), ("IEX", "equities.industrials", 0.5),
    ("IR", "equities.industrials", 0.5), ("ITW", "equities.industrials", 0.5),
    ("J", "equities.industrials", 0.5), ("JBHT", "equities.industrials", 0.5),
    ("JCI", "equities.industrials", 0.5), ("LHX", "equities.industrials", 0.5),
    ("LII", "equities.industrials", 0.5), ("LMT", "equities.industrials", 0.5),
    ("MAS", "equities.industrials", 0.5), ("MMM", "equities.industrials", 0.5),
    ("MSM", "equities.industrials", 0.5), ("NDSN", "equities.industrials", 0.5),
    ("NOC", "equities.industrials", 0.5), ("PWR", "equities.industrials", 0.5),
    ("ROK", "equities.industrials", 0.5), ("RTX", "equities.industrials", 0.5),
    ("SNA", "equities.industrials", 0.5), ("SPCX", "equities.industrials", 0.5),
    ("TTEK", "equities.industrials", 0.5), ("WAB", "equities.industrials", 0.5),
    ("WM", "equities.industrials", 0.5), ("XPO", "equities.industrials", 0.5),
    ("XRX", "equities.industrials", 0.5),
    # healthcare (41)
    ("ABBV", "equities.healthcare", 0.5), ("ABT", "equities.healthcare", 0.5),
    ("ALGN", "equities.healthcare", 0.5), ("AMGN", "equities.healthcare", 0.5),
    ("BIIB", "equities.healthcare", 0.5), ("BMY", "equities.healthcare", 0.5),
    ("BSX", "equities.healthcare", 0.5), ("CAH", "equities.healthcare", 0.5),
    ("CI", "equities.healthcare", 0.5), ("COO", "equities.healthcare", 0.5),
    ("COR", "equities.healthcare", 0.5), ("CRL", "equities.healthcare", 0.5),
    ("CVS", "equities.healthcare", 0.5), ("DHR", "equities.healthcare", 0.5),
    ("DVA", "equities.healthcare", 0.5), ("DXCM", "equities.healthcare", 0.5),
    ("GEHC", "equities.healthcare", 0.5), ("GILD", "equities.healthcare", 0.5),
    ("HCA", "equities.healthcare", 0.5), ("HSIC", "equities.healthcare", 0.5),
    ("HUM", "equities.healthcare", 0.5), ("IDXX", "equities.healthcare", 0.5),
    ("INCY", "equities.healthcare", 0.5), ("ISRG", "equities.healthcare", 0.5),
    ("JNJ", "equities.healthcare", 0.5), ("LH", "equities.healthcare", 0.5),
    ("LLY", "equities.healthcare", 0.5), ("MCK", "equities.healthcare", 0.5),
    ("MDT", "equities.healthcare", 0.5), ("MRK", "equities.healthcare", 0.5),
    ("MTD", "equities.healthcare", 0.5), ("PFE", "equities.healthcare", 0.5),
    ("PODD", "equities.healthcare", 0.5), ("REGN", "equities.healthcare", 0.5),
    ("RMD", "equities.healthcare", 0.5), ("SYK", "equities.healthcare", 0.5),
    ("TMO", "equities.healthcare", 0.5), ("UNH", "equities.healthcare", 0.5),
    ("VRTX", "equities.healthcare", 0.5), ("XRAY", "equities.healthcare", 0.5),
    ("ZTS", "equities.healthcare", 0.5),
    # financials (47)
    ("ACGL", "equities.financials", 0.5), ("AFL", "equities.financials", 0.5),
    ("AIG", "equities.financials", 0.5), ("ALL", "equities.financials", 0.5),
    ("AMP", "equities.financials", 0.5), ("AON", "equities.financials", 0.5),
    ("AXP", "equities.financials", 0.5), ("BAC", "equities.financials", 0.5),
    ("BEN", "equities.financials", 0.5), ("BLK", "equities.financials", 0.5),
    ("BX", "equities.financials", 0.5), ("CB", "equities.financials", 0.5),
    ("CBOE", "equities.financials", 0.5), ("CFG", "equities.financials", 0.5),
    ("CINF", "equities.financials", 0.5), ("CME", "equities.financials", 0.5),
    ("COF", "equities.financials", 0.5), ("COIN", "equities.financials", 0.5),
    ("EG", "equities.financials", 0.5), ("ERIE", "equities.financials", 0.5),
    ("FDS", "equities.financials", 0.5), ("FITB", "equities.financials", 0.5),
    ("GS", "equities.financials", 0.5), ("HIG", "equities.financials", 0.5),
    ("ICE", "equities.financials", 0.5), ("JPM", "equities.financials", 0.5),
    ("KEY", "equities.financials", 0.5), ("KKR", "equities.financials", 0.5),
    ("L", "equities.financials", 0.5), ("MA", "equities.financials", 0.5),
    ("MET", "equities.financials", 0.5), ("MSCI", "equities.financials", 0.5),
    ("MTB", "equities.financials", 0.5), ("NDAQ", "equities.financials", 0.5),
    ("PNC", "equities.financials", 0.5), ("PRU", "equities.financials", 0.5),
    ("PYPL", "equities.financials", 0.5), ("RJF", "equities.financials", 0.5),
    ("SCHW", "equities.financials", 0.5), ("SOFI", "equities.financials", 0.5),
    ("SYF", "equities.financials", 0.5), ("TROW", "equities.financials", 0.5),
    ("TRV", "equities.financials", 0.5), ("UNM", "equities.financials", 0.5),
    ("V", "equities.financials", 0.5), ("WFC", "equities.financials", 0.5),
    ("WRB", "equities.financials", 0.5),
]

# One entry per sector: the strategist reads each sector from ALL its stocks
# (data_ingestion.build_macro_from_snapshots), and judges structural tailwinds
# itself, so there is no hand-picked basket and no hand-set score per sector.
THEMES: List[Tuple[str, List[str], float]] = [(sector, [], 0.0) for sector in SECTORS]

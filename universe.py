"""
universe.py
===========
The single, shared candidate universe: real S&P 500 constituents per sector
(plus a small number of deliberate, explicitly-flagged non-S&P-500
exceptions — see below), so the analyst reasoner picks "top opportunities"
from a real pool, not a hand-picked shortlist. Used by vertex_deploy.py's
set_up() (bundled via extra_packages — see build_and_deploy()) to feed the
deployed agent's live ingestion, and by web_ui.py/simulate.py for local
pricing, buy/sell deltas, and historical simulation. ONE file, not two
copies to keep in sync.

PROVENANCE, stated plainly: this list started from a web-fetched summary of
Wikipedia's S&P 500 constituent table, which turned out to contain real
errors — hallucinated tickers, cross-sector duplicates (e.g. "V" appearing
under both Industrials and Financials), and at least one glaring omission
(JPM missing from Financials entirely). Every symbol below was cross-checked
against Alpaca's actual tradable-assets list (killing the fabricated
tickers), duplicates were resolved to one correct GICS sector each, and a
handful of obviously-missing mega-caps (GOOGL, META, JPM) were added back by
hand. This is NOT a guarantee of GICS-perfect sector accuracy for all 224
names — treat it as a real, broad, spot-checked candidate pool, not an
audited classification.

EXPLICIT NON-S&P-500 EXCEPTIONS: SPCX (Space Exploration Technologies Corp,
i.e. SpaceX) is real and tradable (confirmed live against Alpaca's own
asset list) but is NOT an S&P 500 constituent — S&P's mandatory 12-month
post-IPO seasoning period means it isn't index-eligible until ~June 2027 at
the earliest, despite already being large enough to warrant fast-tracked
inclusion in the Nasdaq-100 and Russell 1000. Added anyway, by explicit
request, rather than waiting for index eligibility — flagged here (and at
its entry below) so this is a documented, deliberate departure from the
"S&P 500 only" rule everything else in this file follows, not an
inconsistency to "fix" later.
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

# (symbol, bucket_path, quality_score) per data_ingestion.UniverseConfig.
EQUITIES: List[Tuple[str, str, float]] = [
    # tech_ai (59)
    ("AAPL", "equities.tech_ai", 0.5), ("ACN", "equities.tech_ai", 0.5),
    ("ADBE", "equities.tech_ai", 0.5), ("ADI", "equities.tech_ai", 0.5),
    ("AKAM", "equities.tech_ai", 0.5), ("AMAT", "equities.tech_ai", 0.5),
    ("AMD", "equities.tech_ai", 0.5), ("APH", "equities.tech_ai", 0.5),
    ("APP", "equities.tech_ai", 0.5), ("AVGO", "equities.tech_ai", 0.5),
    ("CDNS", "equities.tech_ai", 0.5), ("CDW", "equities.tech_ai", 0.5),
    ("COHR", "equities.tech_ai", 0.5), ("CRWD", "equities.tech_ai", 0.5),
    ("CSCO", "equities.tech_ai", 0.5), ("CTSH", "equities.tech_ai", 0.5),
    ("DDOG", "equities.tech_ai", 0.5), ("DELL", "equities.tech_ai", 0.5),
    ("FAST", "equities.tech_ai", 0.5), ("FFIV", "equities.tech_ai", 0.5),
    ("FICO", "equities.tech_ai", 0.5), ("FLEX", "equities.tech_ai", 0.5),
    ("FTNT", "equities.tech_ai", 0.5), ("GDDY", "equities.tech_ai", 0.5),
    ("GLW", "equities.tech_ai", 0.5), ("GOOGL", "equities.tech_ai", 0.5),
    ("HPE", "equities.tech_ai", 0.5), ("HPQ", "equities.tech_ai", 0.5),
    ("IBM", "equities.tech_ai", 0.5), ("INTC", "equities.tech_ai", 0.5),
    ("INTU", "equities.tech_ai", 0.5), ("IT", "equities.tech_ai", 0.5),
    ("KEYS", "equities.tech_ai", 0.5), ("KLAC", "equities.tech_ai", 0.5),
    ("LITE", "equities.tech_ai", 0.5), ("LRCX", "equities.tech_ai", 0.5),
    ("META", "equities.tech_ai", 0.5), ("MRVL", "equities.tech_ai", 0.5),
    ("MSFT", "equities.tech_ai", 0.5), ("NFLX", "equities.tech_ai", 0.5),
    ("NTAP", "equities.tech_ai", 0.5), ("NVDA", "equities.tech_ai", 0.5),
    ("ORCL", "equities.tech_ai", 0.5), ("PAYC", "equities.tech_ai", 0.5),
    ("PYPL", "equities.tech_ai", 0.5), ("QCOM", "equities.tech_ai", 0.5),
    ("SMCI", "equities.tech_ai", 0.5), ("SNOW", "equities.tech_ai", 0.5),
    ("SNPS", "equities.tech_ai", 0.5), ("STX", "equities.tech_ai", 0.5),
    ("SWKS", "equities.tech_ai", 0.5), ("TMUS", "equities.tech_ai", 0.5),
    ("TSLA", "equities.tech_ai", 0.5),   # S&P 500 constituent, confirmed tradable on Alpaca —
                                        # GICS classifies it Consumer Discretionary (no such
                                        # sector exists in this app), placed here as the closest
                                        # fit given this bucket's own AI/autonomy/robotics names
    ("TTEK", "equities.tech_ai", 0.5), ("TYL", "equities.tech_ai", 0.5),
    ("VRSN", "equities.tech_ai", 0.5), ("WDAY", "equities.tech_ai", 0.5),
    ("WDC", "equities.tech_ai", 0.5), ("XRX", "equities.tech_ai", 0.5),
    # energy (16)
    ("AES", "equities.energy", 0.5), ("APA", "equities.energy", 0.5),
    ("BKR", "equities.energy", 0.5), ("COP", "equities.energy", 0.5),
    ("CVX", "equities.energy", 0.5), ("DVN", "equities.energy", 0.5),
    ("EOG", "equities.energy", 0.5), ("EQT", "equities.energy", 0.5),
    ("EXE", "equities.energy", 0.5), ("HAL", "equities.energy", 0.5),
    ("KMI", "equities.energy", 0.5), ("MPC", "equities.energy", 0.5),
    ("OKE", "equities.energy", 0.5), ("SLB", "equities.energy", 0.5),
    ("TRGP", "equities.energy", 0.5), ("XOM", "equities.energy", 0.5),
    # industrials (57)
    ("ADP", "equities.industrials", 0.5), ("AME", "equities.industrials", 0.5),
    ("AOS", "equities.industrials", 0.5), ("AXON", "equities.industrials", 0.5),
    ("BA", "equities.industrials", 0.5), ("BLDR", "equities.industrials", 0.5),
    ("BR", "equities.industrials", 0.5), ("CARR", "equities.industrials", 0.5),
    ("CAT", "equities.industrials", 0.5), ("CHRW", "equities.industrials", 0.5),
    ("CMI", "equities.industrials", 0.5), ("CPRT", "equities.industrials", 0.5),
    ("CSX", "equities.industrials", 0.5), ("DD", "equities.industrials", 0.5),
    ("DE", "equities.industrials", 0.5), ("DOV", "equities.industrials", 0.5),
    ("EFX", "equities.industrials", 0.5), ("EME", "equities.industrials", 0.5),
    ("EMR", "equities.industrials", 0.5), ("ETN", "equities.industrials", 0.5),
    ("EXPD", "equities.industrials", 0.5), ("FDX", "equities.industrials", 0.5),
    ("FERG", "equities.industrials", 0.5), ("FTV", "equities.industrials", 0.5),
    ("GD", "equities.industrials", 0.5), ("GE", "equities.industrials", 0.5),
    ("GNRC", "equities.industrials", 0.5), ("HII", "equities.industrials", 0.5),
    ("HON", "equities.industrials", 0.5), ("HWM", "equities.industrials", 0.5),
    ("IEX", "equities.industrials", 0.5), ("IR", "equities.industrials", 0.5),
    ("ITW", "equities.industrials", 0.5), ("J", "equities.industrials", 0.5),
    ("JBHT", "equities.industrials", 0.5), ("JCI", "equities.industrials", 0.5),
    ("LDOS", "equities.industrials", 0.5), ("LHX", "equities.industrials", 0.5),
    ("LII", "equities.industrials", 0.5), ("LMT", "equities.industrials", 0.5),
    ("MAS", "equities.industrials", 0.5), ("MLM", "equities.industrials", 0.5),
    ("MMM", "equities.industrials", 0.5), ("MSM", "equities.industrials", 0.5),
    ("NDSN", "equities.industrials", 0.5), ("NOC", "equities.industrials", 0.5),
    ("PAYX", "equities.industrials", 0.5), ("PWR", "equities.industrials", 0.5),
    ("ROK", "equities.industrials", 0.5), ("RTX", "equities.industrials", 0.5),
    ("SNA", "equities.industrials", 0.5),
    ("SPCX", "equities.industrials", 0.5),   # Space Exploration Technologies Corp (SpaceX) —
                                             # IPO'd June 12, 2026 on Nasdaq, confirmed real and
                                             # tradable on Alpaca. NOT an S&P 500 constituent
                                             # (S&P's mandatory 12-month post-IPO seasoning period
                                             # means it can't be added before ~June 2027 at the
                                             # earliest) — this is a DELIBERATE, explicit exception
                                             # to this universe's otherwise S&P-500-only design,
                                             # not an oversight. See the module docstring.
    ("STLD", "equities.industrials", 0.5),
    ("VFC", "equities.industrials", 0.5), ("WAB", "equities.industrials", 0.5),
    ("WM", "equities.industrials", 0.5), ("XPO", "equities.industrials", 0.5),
    # healthcare (42)
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
    ("JNJ", "equities.healthcare", 0.5), ("KVUE", "equities.healthcare", 0.5),
    ("LH", "equities.healthcare", 0.5), ("LLY", "equities.healthcare", 0.5),
    ("MCK", "equities.healthcare", 0.5), ("MDT", "equities.healthcare", 0.5),
    ("MRK", "equities.healthcare", 0.5), ("MTD", "equities.healthcare", 0.5),
    ("PFE", "equities.healthcare", 0.5), ("PODD", "equities.healthcare", 0.5),
    ("REGN", "equities.healthcare", 0.5), ("RMD", "equities.healthcare", 0.5),
    ("SYK", "equities.healthcare", 0.5), ("TMO", "equities.healthcare", 0.5),
    ("UNH", "equities.healthcare", 0.5), ("VRTX", "equities.healthcare", 0.5),
    ("XRAY", "equities.healthcare", 0.5), ("ZTS", "equities.healthcare", 0.5),
    # financials (51)
    ("ACGL", "equities.financials", 0.5), ("AFL", "equities.financials", 0.5),
    ("AIG", "equities.financials", 0.5), ("ALL", "equities.financials", 0.5),
    ("AMP", "equities.financials", 0.5), ("AON", "equities.financials", 0.5),
    ("AXP", "equities.financials", 0.5), ("BAC", "equities.financials", 0.5),
    ("BEN", "equities.financials", 0.5), ("BLK", "equities.financials", 0.5),
    ("BX", "equities.financials", 0.5), ("CB", "equities.financials", 0.5),
    ("CBOE", "equities.financials", 0.5), ("CFG", "equities.financials", 0.5),
    ("CINF", "equities.financials", 0.5), ("CME", "equities.financials", 0.5),
    ("COF", "equities.financials", 0.5), ("COIN", "equities.financials", 0.5),
    ("CPAY", "equities.financials", 0.5), ("EG", "equities.financials", 0.5),
    ("ERIE", "equities.financials", 0.5), ("FDS", "equities.financials", 0.5),
    ("FIS", "equities.financials", 0.5), ("FITB", "equities.financials", 0.5),
    ("GS", "equities.financials", 0.5), ("HIG", "equities.financials", 0.5),
    ("ICE", "equities.financials", 0.5), ("JKHY", "equities.financials", 0.5),
    ("JPM", "equities.financials", 0.5), ("KEY", "equities.financials", 0.5),
    ("KKR", "equities.financials", 0.5), ("L", "equities.financials", 0.5),
    ("MA", "equities.financials", 0.5), ("MET", "equities.financials", 0.5),
    ("MSCI", "equities.financials", 0.5), ("MTB", "equities.financials", 0.5),
    ("NDAQ", "equities.financials", 0.5), ("PNC", "equities.financials", 0.5),
    ("PRU", "equities.financials", 0.5), ("RJF", "equities.financials", 0.5),
    ("SCHW", "equities.financials", 0.5), ("SLG", "equities.financials", 0.5),
    ("SOFI", "equities.financials", 0.5), ("SYF", "equities.financials", 0.5),
    ("TROW", "equities.financials", 0.5), ("TRV", "equities.financials", 0.5),
    ("UNM", "equities.financials", 0.5), ("V", "equities.financials", 0.5),
    ("WFC", "equities.financials", 0.5), ("WRB", "equities.financials", 0.5),
    ("XYZ", "equities.financials", 0.5),
]

# Theme-level snapshots feed strategist_reasoning_node, which produces the
# theme_tilts multi-sector recommendations lock onto (web_ui.py).
# Deliberately kept to a SMALL flagship subset per sector, not the full
# 200+-name EQUITIES list above: data_ingestion.build_macro_snapshot
# re-fetches price history per constituent, so using the full list here
# would roughly double live-ingestion latency for no real signal benefit
# (theme momentum is already just an average). secular_score is a rough,
# adjustable placeholder for "structural tailwind," same honesty caveat as
# the ticker list above.
THEMES: List[Tuple[str, List[str], float]] = [
    ("tech_ai",     ["NVDA", "MSFT", "GOOGL", "META", "AVGO"], 0.3),
    ("energy",      ["XOM", "CVX"],                           0.0),
    ("industrials", ["CAT", "HON"],                           0.0),
    ("healthcare",  ["JNJ", "UNH"],                           0.1),
    ("financials",  ["JPM", "BAC"],                           0.0),
]

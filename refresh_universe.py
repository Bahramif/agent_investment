"""
refresh_universe.py
===================
Re-files universe.py's stocks by the data provider's sector
(universe.PROVIDER_SECTORS). Prints every change and rewrites nothing unless
--write is passed. A stock the provider has no data for stays where it is.

Run:  python refresh_universe.py [--write]
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

from universe import EQUITIES, PROVIDER_SECTORS, SECTORS


def provider_sector(symbol: str):
    try:
        return yf.Ticker(symbol).get_info().get("sector")
    except Exception as e:
        print(f"  no data for {symbol}: {e}")
        return None


def main(write: bool) -> None:
    with ThreadPoolExecutor(max_workers=6) as pool:
        found = list(pool.map(provider_sector, [s for s, _, _ in EQUITIES]))
    groups = {sector: [] for sector in SECTORS}
    for (symbol, path, _), sector in zip(EQUITIES, found):
        filed = path.split(".")[-1]
        target = PROVIDER_SECTORS.get(sector) if sector else filed
        if target != filed:
            print(f"  {symbol}: {filed} -> {target or 'removed'} (provider says {sector})")
        if target:
            groups[target].append(symbol)
    if not write:
        print("Dry run. Pass --write to apply.")
        return
    lines = ["EQUITIES: List[Tuple[str, str, float]] = ["]
    for sector, symbols in groups.items():
        lines.append(f"    # {sector} ({len(symbols)})")
        symbols = sorted(symbols)
        for i in range(0, len(symbols), 2):
            lines.append("    " + " ".join(f'("{s}", "equities.{sector}", 0.5),' for s in symbols[i:i + 2]))
    lines.append("]")
    with open("universe.py") as fh:
        src = fh.read()
    with open("universe.py", "w") as fh:
        fh.write(re.sub(r"EQUITIES: List\[Tuple\[str, str, float\]\] = \[.*?^\]", "\n".join(lines), src,
                        count=1, flags=re.S | re.M))
    print("universe.py updated.")


if __name__ == "__main__":
    main("--write" in sys.argv)

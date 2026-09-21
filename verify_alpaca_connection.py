"""
verify_alpaca_connection.py
=============================
Run this on YOUR machine (not this sandbox — it needs real network access
and real keys, neither of which exist here). It checks two things in order:

  1. Do your ALPACA_API_KEY / ALPACA_SECRET_KEY actually work at all?
  2. Does THIS PROJECT'S OWN get_broker_client() wrapper work against them?

Step 2 matters more than it looks: it's the difference between "my Alpaca
keys are valid" and "the actual code path this system will run in
production works end to end." Testing only the raw alpaca-py client can
pass while a bug in get_broker_client() itself goes unnoticed.

Usage:
    pip install alpaca-py
    export ALPACA_API_KEY=your_key_id
    export ALPACA_SECRET_KEY=your_secret
    python3 verify_alpaca_connection.py
"""

import os
import sys


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("✗ ALPACA_API_KEY / ALPACA_SECRET_KEY are not set in your environment.")
        print("  export ALPACA_API_KEY=...")
        print("  export ALPACA_SECRET_KEY=...")
        sys.exit(1)

    print("=" * 70)
    print("STEP 1: raw connection — does alpaca-py itself accept your keys?")
    print("=" * 70)
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        print("✗ alpaca-py is not installed. Run: pip install alpaca-py")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=True)
    try:
        account = client.get_account()
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("  Double-check you copied both the Key ID and Secret correctly,")
        print("  and that you generated them from the PAPER dashboard, not live.")
        sys.exit(1)

    print(f"✓ Connected. Account status: {account.status}")
    print(f"  Buying power: ${float(account.buying_power):,.2f}")
    print(f"  Portfolio value: ${float(account.portfolio_value):,.2f}")

    print("\n" + "=" * 70)
    print("STEP 2: this project's own get_broker_client() — the real code path")
    print("=" * 70)
    try:
        from execution_workflow import Order, Side, get_broker_client
    except ImportError as e:
        print(f"(Skipping — execution_workflow.py not found alongside this script: {e})")
        print("  Copy this file into the same directory as this project's other .py files.")
        return

    try:
        broker = get_broker_client()
    except Exception as e:
        print(f"✗ get_broker_client() raised: {e}")
        print("  Check ALPACA_PAPER is unset or 'true', and that execution_workflow.py's")
        print("  get_broker_client() hasn't drifted from what this script expects.")
        sys.exit(1)

    print("✓ get_broker_client() constructed an AlpacaBrokerClient successfully.")

    print("\n" + "=" * 70)
    print("STEP 3: a real (paper) position query through this project's wrapper")
    print("=" * 70)
    positions = broker.get_current_positions(total_capital=float(account.portfolio_value) or 1.0)
    if positions:
        print(f"✓ Current positions (as weight of portfolio): {positions}")
    else:
        print("✓ No open positions yet — expected for a fresh paper account. This still")
        print("  confirms get_current_positions() completed without error.")

    print("\n" + "=" * 70)
    print("STEP 4 (OPTIONAL, COMMENTED OUT): a real paper order")
    print("=" * 70)
    print("Uncomment the block below to place one real (paper) trade — a tiny")
    print("$10 NVDA buy — and watch it flow through submit_and_await(), including")
    print("the actual poll-until-filled loop, for real. Left commented by default")
    print("so running this script doesn't unexpectedly place an order.")
    print("""
    # test_order = Order("NVDA", Side.BUY, notional=10.0, client_order_id="verify-test-001")
    # fill = broker.submit_and_await(test_order)
    # print(f"Result: {fill.result.value}, filled_notional=${fill.filled_notional:.2f}")
    """)

    print("=" * 70)
    print("All checks passed. Your keys work, and this project's own broker")
    print("wrapper works against them — not just Alpaca's raw SDK.")
    print("=" * 70)


if __name__ == "__main__":
    main()

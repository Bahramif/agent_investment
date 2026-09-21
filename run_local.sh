#!/bin/bash
# run_local.sh
# Starts the Meridian webapp on your own machine at http://127.0.0.1:5050
# Run this yourself in a terminal (Terminal.app, iTerm, etc.) — it must run
# on the same machine as the browser you'll use to open the app.
set -e
cd "$(dirname "$0")"
exec .venv/bin/python3 web_ui.py

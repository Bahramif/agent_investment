# Runs web_ui.py as a real webapp on Cloud Run instead of Flask's local
# development server (which explicitly warns against production use).
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web_ui.py universe.py simulate.py mandate_allocator.py analyst_reasoning.py \
     claude_reasoning.py execution_workflow.py alpaca_clients.py data_ingestion.py \
     strategist.py macro_reasoning.py portfolio_store.py fundamentals.py proactive_recal.py ./

# Day-trading capability. The SAME image serves the web tab (daytrade_ui.py) and
# the Cloud Run Job that trades (daytrade_runner.py — see its docstring for the
# `gcloud run jobs create ... --command python --args daytrade_runner.py`).
# mandate.yaml carries the day_trading: config; promotion_gate.py is imported by
# daytrade_backtest.py (the runner reuses its SimBroker for the shadow engine).
COPY daytrade_config.py intraday_data.py daytrade_strategies.py daytrade_risk.py \
     daytrade_engine.py daytrade_backtest.py daytrade_broker.py daytrade_store.py \
     daytrade_playbook.py daytrade_runner.py daytrade_ui.py promotion_gate.py \
     mandate.yaml ./
# Failure-learning loop: imported by the engine (journal, lessons), the runner's
# post-close hook and the dashboard's Learning page.
COPY daytrade_journal.py daytrade_lessons.py daytrade_attribution.py daytrade_reflection.py \
     daytrade_learning.py ./

ENV PORT=8080
EXPOSE 8080

# --timeout 900: real recommendation requests have taken up to ~5-6 minutes
# under heavy Alpaca API load (see vertex_deploy.py's ingestion concurrency
# comment) — gunicorn's default 30s worker timeout would kill those requests
# mid-flight. --workers 1 --threads 8 --worker-class gthread: a personal
# single-user tool, no need for multiple worker processes, but gthread (not
# the default sync class, which ignores --threads entirely) is required so
# /recommend_stream's long-lived SSE connection doesn't block every other
# request for its whole multi-minute duration.
CMD exec gunicorn --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 8 --timeout 900 web_ui:app

"""
whatsapp_notify.py
===================
Optional WhatsApp notifications for portfolio events, through Meta's
WhatsApp Cloud API. Off unless the owner turns them on for a portfolio and
gives a number (portfolio page).

What is sent:
  * every recalibration — scheduled, mid-day news watch, or manual — with the
    trades placed and (for the news watch) Gemini's reason;
  * news-watch FLAGS — Gemini wanted to recalibrate but a rule stopped the
    trade (already recalibrated today, orders still filling, monthly cap,
    market closed). Dry runs never notify.

HOW META SENDS IT. A message the business starts must use a TEMPLATE that
Meta has approved. One generic template carries every event, so it is
approved once (WhatsApp Manager -> Message templates; category UTILITY):

    name:      meridian_portfolio_update        (config: notifications.whatsapp.template_name)
    language:  English (US) — en_US
    body:      Meridian portfolio update for {{1}}.

               {{2}}

               Details: {{3}}

               Automated notice from your Meridian paper-trading app.

  {{1}} portfolio name, {{2}} one-line headline, {{3}} trades / reason.
  Template parameters may not contain newlines or tabs, so clean_param()
  flattens them and caps their length.

  POST https://graph.facebook.com/{api_version}/{phone_number_id}/messages
  Authorization: Bearer {access token}
  {"messaging_product": "whatsapp", "recipient_type": "individual", "to": "<digits>",
   "type": "template", "template": {"name": ..., "language": {"code": "en_US"},
   "components": [{"type": "body", "parameters": [{"type": "text", "text": ...} x3]}]}}

Credentials live in Secret Manager (web_ui reads them): `whatsapp-access-token`
(a System User token with whatsapp_business_messaging permission) and
`whatsapp-phone-number-id` (the sender's Phone number ID, not the number).

NEVER RAISES into a recalibration: notify() returns an outcome dict
({"sent": bool, ...}) that the caller logs on the portfolio; a WhatsApp
outage can't block or undo a trade. Stdlib only (urllib).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

PARAM_MAX = 900          # stay well under Meta's per-parameter limit
TRIGGER_LABEL = {"scheduled": "📅 Scheduled recalibration", "proactive": "📰 Mid-day news-watch recalibration",
                 "manual": "👆 Manual recalibration"}


@dataclass
class WhatsAppConfig:
    enabled: bool = True
    api_version: str = "v26.0"
    template_name: str = "meridian_portfolio_update"
    language_code: str = "en_US"


def config_from_dict(raw: Optional[Dict[str, Any]]) -> WhatsAppConfig:
    raw = dict(raw or {})
    unknown = set(raw) - set(WhatsAppConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"notifications.whatsapp: unknown key(s) {sorted(unknown)}")
    cfg = WhatsAppConfig(**raw)
    if not re.fullmatch(r"v\d+\.\d+", cfg.api_version):
        raise ValueError("notifications.whatsapp.api_version must look like 'v26.0'")
    if not re.fullmatch(r"[a-z0-9_]{1,512}", cfg.template_name):
        raise ValueError("notifications.whatsapp.template_name: lowercase letters, digits and underscores only")
    return cfg


def load_config(path: str = "mandate.yaml") -> WhatsAppConfig:
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(((data.get("notifications") or {}).get("whatsapp")))


# -----------------------------------------------------------------------------
# Numbers and text
# -----------------------------------------------------------------------------
def normalize_number(raw: str) -> str:
    """International format -> digits only (what the Cloud API's `to` takes).
    Accepts '+1 (416) 555-0100', '0014165550100' is NOT accepted (no IDD
    prefixes — use + and the country code). Raises ValueError."""
    s = re.sub(r"[\s().\-]", "", str(raw or ""))
    if not s.startswith("+"):
        raise ValueError("start the number with + and the country code, e.g. +1 416 555 0100")
    digits = s[1:]
    if not digits.isdigit() or not 8 <= len(digits) <= 15 or digits.startswith("0"):
        raise ValueError("that doesn't look like an international phone number (+, country code, 8-15 digits)")
    return digits


def mask_number(digits: str) -> str:
    return f"+{digits[:2]}…{digits[-3:]}" if digits else ""


def clean_param(text: Any, max_len: int = PARAM_MAX) -> str:
    """Template parameters: no newlines/tabs, no runs of more than 3 spaces, non-empty."""
    s = re.sub(r"[\r\n\t]+", " · ", str(text or "")).strip()
    s = re.sub(r" {4,}", "   ", s)
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s or "-"


# -----------------------------------------------------------------------------
# Messages
# -----------------------------------------------------------------------------
def _trades(orders: List[Dict[str, Any]]) -> str:
    def side(sd: str) -> str:
        items = [f"{o.get('symbol')} {o.get('qty')}" for o in orders if str(o.get("side", "")).upper() == sd]
        return ", ".join(items)
    parts = []
    if side("SELL"):
        parts.append(f"Sold {side('SELL')}")
    if side("BUY"):
        parts.append(f"Bought {side('BUY')}")
    return "; ".join(parts) or "no trades needed — already on target"


def recalibration_message(event: Dict[str, Any]) -> Tuple[str, str]:
    """(headline, details) for one rebalance_history event."""
    orders = event.get("orders") or []
    errors = event.get("errors") or []
    label = TRIGGER_LABEL.get(event.get("trigger") or "", "Recalibration")
    n_buy = sum(1 for o in orders if str(o.get("side", "")).upper() == "BUY")
    n_sell = len(orders) - n_buy
    headline = f"{label}: {n_sell} sell(s), {n_buy} buy(s)" if orders else f"{label}: no changes needed"
    if errors:
        headline += f", {len(errors)} order error(s)"
    details = _trades(orders)
    if event.get("portfolio_value_before"):
        details += f". Portfolio value before: ${float(event['portfolio_value_before']):,.0f}"
    if event.get("reason"):
        details += f". Why: {event['reason']}"
    return headline, details + "."


def flag_message(reason: str, model: str = "") -> Tuple[str, str]:
    """A news-watch verdict of 'recalibrate' that was NOT traded."""
    headline = "📰 News watch suggests recalibrating — NOT traded automatically"
    details = f"{reason}" + (f" (decided by {model})" if model else "") + ". Open the portfolio to recalibrate it yourself."
    return headline, details


def test_message() -> Tuple[str, str]:
    return ("✅ WhatsApp notifications are set up", "You will get a message here when this portfolio recalibrates "
            "or the mid-day news watch flags it.")


def build_payload(to_digits: str, cfg: WhatsAppConfig, portfolio_name: str, headline: str, details: str) -> Dict[str, Any]:
    return {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to_digits, "type": "template",
            "template": {"name": cfg.template_name, "language": {"code": cfg.language_code},
                         "components": [{"type": "body", "parameters": [
                             {"type": "text", "text": clean_param(portfolio_name, 120)},
                             {"type": "text", "text": clean_param(headline, 300)},
                             {"type": "text", "text": clean_param(details)}]}]}}


# -----------------------------------------------------------------------------
# Sending
# -----------------------------------------------------------------------------
HttpFn = Callable[[str, Dict[str, str], bytes], Tuple[int, bytes]]


def _urllib_post(url: str, headers: Dict[str, str], body: bytes) -> Tuple[int, bytes]:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def send(payload: Dict[str, Any], token: str, phone_number_id: str, cfg: WhatsAppConfig,
         http: HttpFn = _urllib_post) -> str:
    """Returns Meta's message id; raises RuntimeError with Meta's error text."""
    url = f"https://graph.facebook.com/{cfg.api_version}/{phone_number_id}/messages"
    status, raw = http(url, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                       json.dumps(payload).encode())
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = {}
    if 200 <= status < 300 and body.get("messages"):
        return str(body["messages"][0].get("id", ""))
    err = body.get("error") or {}
    raise RuntimeError(f"WhatsApp API {status}: {err.get('message') or raw[:200]!r}"
                       + (f" (code {err.get('code')})" if err.get("code") else ""))


def notify(portfolio: Dict[str, Any], headline: str, details: str, cfg: WhatsAppConfig,
           credentials: Callable[[], Optional[Tuple[str, str]]], http: HttpFn = _urllib_post,
           force: bool = False) -> Dict[str, Any]:
    """Send one notification for this portfolio if it has them turned on (or
    `force`, for the test button). NEVER raises; returns the outcome to log."""
    out: Dict[str, Any] = {"headline": clean_param(headline, 300), "sent": False}
    try:
        if not cfg.enabled:
            out["skipped"] = "WhatsApp notifications are disabled in mandate.yaml"
            return out
        if not force and portfolio.get("whatsapp_notify") is not True:
            out["skipped"] = "notifications off for this portfolio"
            return out
        to = portfolio.get("whatsapp_number") or ""
        if not to:
            out["skipped"] = "no WhatsApp number set"
            return out
        out["to"] = mask_number(to)
        creds = credentials()
        if not creds:
            out["error"] = "WhatsApp credentials not configured (whatsapp-access-token / whatsapp-phone-number-id)"
            return out
        token, phone_id = creds
        out["message_id"] = send(build_payload(to, cfg, portfolio.get("name", ""), headline, details),
                                 token, phone_id, cfg, http)
        out["sent"] = True
    except Exception as e:
        out["error"] = str(e)[:300]
    return out


if __name__ == "__main__":
    print("TEST 1: config and number handling")
    assert config_from_dict(None) == WhatsAppConfig()
    for bad in ({"api_vesion": "v26.0"}, {"api_version": "26"}, {"template_name": "Has Spaces"}):
        try:
            config_from_dict(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert normalize_number("+1 (416) 555-0100") == "14165550100"
    assert normalize_number("+44 7911 123456") == "447911123456"
    for bad in ("4165550100", "+0123456789", "+1 416", "+1-416-CALL-NOW", ""):
        try:
            normalize_number(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert mask_number("14165550100") == "+14…100"
    print("✓ needs + and country code; letters / too short / leading 0 rejected; numbers masked in logs")

    print("TEST 2: parameters are flattened and capped (Meta rejects newlines/tabs in template params)")
    p = clean_param("line one\nline two\t\ttabbed      wide", 1000)
    assert "\n" not in p and "\t" not in p and "    " not in p and p.startswith("line one · line two")
    assert len(clean_param("x" * 5000)) == PARAM_MAX and clean_param("") == "-"
    print(f"✓ {p!r}")

    print("TEST 3: messages for each event")
    ev = {"date": "2026-10-12", "trigger": "proactive", "reason": "CCC won a $20B contract\nand raised guidance",
          "orders": [{"symbol": "AAA", "side": "SELL", "qty": 40}, {"symbol": "CCC", "side": "BUY", "qty": 30},
                     {"symbol": "BBB", "side": "BUY", "qty": 5}], "errors": [], "portfolio_value_before": 10234.5}
    h, d = recalibration_message(ev)
    assert h == "📰 Mid-day news-watch recalibration: 1 sell(s), 2 buy(s)"
    assert d.startswith("Sold AAA 40; Bought CCC 30, BBB 5. Portfolio value before: $10,234. Why: CCC won")
    h2, d2 = recalibration_message({"trigger": "scheduled", "orders": [], "errors": []})
    assert h2 == "📅 Scheduled recalibration: no changes needed" and "already on target" in d2
    h3, _ = recalibration_message({"trigger": "manual", "orders": [{"symbol": "A", "side": "BUY", "qty": 1}],
                                   "errors": [{"symbol": "B", "error": "x"}]})
    assert h3.endswith("0 sell(s), 1 buy(s), 1 order error(s)")
    fh, fd = flag_message("Probe at AAA (not traded: already recalibrated today)", "gemini-3.1-pro-preview")
    assert "NOT traded" in fh and "gemini-3.1-pro-preview" in fd
    print(f"✓ {h} | {d[:70]}…")

    print("TEST 4: payload shape and a successful send")
    calls = []
    def ok_http(url, headers, body):
        calls.append((url, headers, json.loads(body)))
        return 200, json.dumps({"messaging_product": "whatsapp", "contacts": [{"input": "1", "wa_id": "1"}],
                                "messages": [{"id": "wamid.ABC"}]}).encode()
    cfg = WhatsAppConfig()
    pf = {"name": "Tech", "whatsapp_notify": True, "whatsapp_number": "14165550100"}
    out = notify(pf, h, d, cfg, lambda: ("TOKEN", "PNID"), ok_http)
    url, headers, body = calls[0]
    assert out == {"headline": h, "sent": True, "to": "+14…100", "message_id": "wamid.ABC"}, out
    assert url == "https://graph.facebook.com/v26.0/PNID/messages" and headers["Authorization"] == "Bearer TOKEN"
    assert body["to"] == "14165550100" and body["type"] == "template"
    assert body["template"]["name"] == "meridian_portfolio_update" and body["template"]["language"] == {"code": "en_US"}
    params = body["template"]["components"][0]["parameters"]
    assert [x["text"] for x in params][:2] == ["Tech", h] and all("\n" not in x["text"] for x in params)
    print("✓ POST graph.facebook.com/v26.0/PNID/messages, template with 3 body params, wamid returned")

    print("TEST 5: never raises — off, no number, no credentials, Meta errors, network failure")
    assert notify({**pf, "whatsapp_notify": False}, h, d, cfg, lambda: ("T", "P"), ok_http)["skipped"]
    assert notify({**pf, "whatsapp_number": ""}, h, d, cfg, lambda: ("T", "P"), ok_http)["skipped"]
    assert notify(pf, h, d, WhatsAppConfig(enabled=False), lambda: ("T", "P"), ok_http)["skipped"]
    assert "credentials" in notify(pf, h, d, cfg, lambda: None, ok_http)["error"]
    meta_err = lambda u, hd, b: (400, json.dumps({"error": {"message": "Template name does not exist in the translation",
                                                            "code": 132001}}).encode())
    r = notify(pf, h, d, cfg, lambda: ("T", "P"), meta_err)
    assert r["sent"] is False and "132001" in r["error"] and "Template name" in r["error"]
    def down(u, hd, b):
        raise OSError("network unreachable")
    assert "unreachable" in notify(pf, h, d, cfg, lambda: ("T", "P"), down)["error"]
    assert notify({**pf, "whatsapp_notify": False}, *test_message(), cfg, lambda: ("T", "P"), ok_http, force=True)["sent"]
    print("✓ skips/errors come back as data; Meta's error text and code kept for the log; test send forces")
    print("\nAll whatsapp_notify tests passed.")

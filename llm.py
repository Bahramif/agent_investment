"""
llm.py
======
Which model provider the firm's AI steps use, in ONE place.

Every AI call site in this project (claude_reasoning's strategist and analyst
desks, the day-trading playbook and post-close reflection, the portfolio news
watch, the chat box) was written against the Anthropic SDK shape:

    client.messages.create(model=..., max_tokens=..., tools=[...],
                           tool_choice={"type": "tool", "name": ...},
                           system=..., messages=[{"role": "user", "content": ...}])
    -> response.content = [blocks with .type "tool_use" (.name, .input) / "text" (.text)]
       response.stop_reason

GeminiClaudeShim exposes exactly that surface on top of Gemini on Vertex AI
(google-genai SDK), so every caller — and its clamping, fallbacks and tests —
works unchanged. Only client construction moves here.

  * Tools become Gemini function declarations (same JSON schema); a forced
    tool_choice becomes function-calling mode ANY restricted to that name.
  * THINKING is on for every call: a thinking budget per tier (strong 2048,
    fast 1024 tokens, overridable). Thought parts are never returned as
    output; their token count is logged so cost stays visible.
  * Claude model IDs passed by callers are mapped by tier: anything "opus"
    or "sonnet" -> the strong Gemini model, anything else (haiku) -> fast.
  * A blocked prompt/response maps to stop_reason "refusal" and a cut-off to
    "max_tokens" — callers already treat both as failures and fall back.

Provider selection (environment, read at call time):
  MERIDIAN_LLM_PROVIDER   "gemini" (default) | "anthropic"
  MERIDIAN_GEMINI_STRONG  default "gemini-3.1-pro-preview"
  MERIDIAN_GEMINI_FAST    default "gemini-3.8-flash"
  MERIDIAN_GEMINI_THINKING_STRONG / _FAST   thinking budgets (tokens)
  MERIDIAN_GEMINI_LOCATION  default "global"

Why Gemini by default: this project's Vertex quota for Claude is effectively
zero (every Claude call returned 429), while Gemini on the same project,
same IAM, works — verified with a forced function call + thinking on
gemini-3.1-pro-preview before this was written.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional

DEFAULT_STRONG = "gemini-3.1-pro-preview"
DEFAULT_FAST = "gemini-3.8-flash"
MAX_OUTPUT_TOKENS = 65536        # Gemini's per-request output limit (thinking counts against it)


def provider() -> str:
    p = os.environ.get("MERIDIAN_LLM_PROVIDER", "gemini").strip().lower()
    return p if p in ("gemini", "anthropic") else "gemini"


def gemini_model_for(claude_model: str) -> str:
    """Tier mapping: the callers' Claude IDs encode how much reasoning they want."""
    m = (claude_model or "").lower()
    if m.startswith("gemini"):
        return claude_model
    strong = "opus" in m or "sonnet" in m
    return os.environ.get("MERIDIAN_GEMINI_STRONG" if strong else "MERIDIAN_GEMINI_FAST",
                          DEFAULT_STRONG if strong else DEFAULT_FAST)


def thinking_budget_for(gemini_model: str) -> int:
    fast = gemini_model == os.environ.get("MERIDIAN_GEMINI_FAST", DEFAULT_FAST)
    key, default = ("MERIDIAN_GEMINI_THINKING_FAST", 1024) if fast else ("MERIDIAN_GEMINI_THINKING_STRONG", 2048)
    try:
        return max(0, int(os.environ.get(key, default)))
    except ValueError:
        return default


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        t = b.get("text") if isinstance(b, dict) else getattr(b, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


class GeminiClaudeShim:
    """Anthropic-shaped client backed by Gemini. `genai_client` is injectable
    for tests; otherwise a Vertex AI google-genai client is built lazily."""

    def __init__(self, project: Optional[str] = None, location: Optional[str] = None, genai_client: Any = None):
        self.project = project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or os.environ.get(
            "MERIDIAN_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("MERIDIAN_GEMINI_LOCATION", "global")
        self._genai = genai_client
        self._lock = threading.Lock()
        self.messages = self            # so client.messages.create(...) works

    def _client(self) -> Any:
        """ONE google-genai client per shim, created under a lock. Callers
        fan out across threads (desk batches); without the lock each thread
        built its own client and the losers were garbage-collected — which
        closes their HTTP connection mid-request ("client has been closed")."""
        if self._genai is None:
            with self._lock:
                if self._genai is None:
                    from google import genai
                    self._genai = genai.Client(vertexai=True, project=self.project, location=self.location)
        return self._genai

    def create(self, *, model: str, max_tokens: int, messages: List[Dict[str, Any]],
               tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[Dict[str, Any]] = None,
               system: Any = None, **_ignored: Any) -> Any:
        from google.genai import types

        gmodel = gemini_model_for(model)
        budget = thinking_budget_for(gmodel)
        contents = [types.Content(role="model" if m.get("role") == "assistant" else "user",
                                  parts=[types.Part(text=_text_of(m.get("content")))])
                    for m in messages if m.get("role") in ("user", "assistant")]
        cfg: Dict[str, Any] = {
            # thinking draws from the same output budget; never ask past the model's limit (400 otherwise)
            "max_output_tokens": min(int(max_tokens) + budget, MAX_OUTPUT_TOKENS - 1),
            "thinking_config": types.ThinkingConfig(thinking_budget=budget, include_thoughts=False),
        }
        sys_text = _text_of(system) if system else ""
        if sys_text:
            cfg["system_instruction"] = sys_text
        if tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t.get("description", ""),
                                          parameters_json_schema=t.get("input_schema") or {"type": "object"})
                for t in tools])]
            choice = (tool_choice or {}).get("type")
            if choice == "tool":
                fcc = types.FunctionCallingConfig(mode="ANY", allowed_function_names=[tool_choice["name"]])
            elif choice == "any":
                fcc = types.FunctionCallingConfig(mode="ANY")
            else:
                fcc = types.FunctionCallingConfig(mode="AUTO")
            cfg["tool_config"] = types.ToolConfig(function_calling_config=fcc)
            # declarations only (no Python callables): the caller reads the call, nothing auto-executes
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        r = self._client().models.generate_content(model=gmodel, contents=contents,
                                                   config=types.GenerateContentConfig(**cfg))
        return _to_claude_shape(r, gmodel)


def _to_claude_shape(r: Any, gmodel: str) -> Any:
    blocks: List[Any] = []
    finish = ""
    cands = getattr(r, "candidates", None) or []
    if cands:
        c0 = cands[0]
        finish = str(getattr(getattr(c0, "finish_reason", None), "name", getattr(c0, "finish_reason", "")) or "")
        for i, part in enumerate(getattr(getattr(c0, "content", None), "parts", None) or []):
            if getattr(part, "thought", False):
                continue                                    # reasoning stays internal
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                blocks.append(NS(type="tool_use", id=f"gemini_call_{i}", name=fc.name, input=dict(fc.args or {})))
            elif getattr(part, "text", None):
                blocks.append(NS(type="text", text=part.text))
    blocked = getattr(getattr(r, "prompt_feedback", None), "block_reason", None)
    if blocked or finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
        stop = "refusal"
    elif any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    elif finish == "MAX_TOKENS":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    u = getattr(r, "usage_metadata", None)
    usage = NS(input_tokens=getattr(u, "prompt_token_count", 0) or 0,
               output_tokens=getattr(u, "candidates_token_count", 0) or 0,
               thinking_tokens=getattr(u, "thoughts_token_count", 0) or 0)
    print(f"[llm] {gmodel}: stop={stop} in={usage.input_tokens} thinking={usage.thinking_tokens} "
          f"out={usage.output_tokens}", flush=True)
    return NS(content=blocks, stop_reason=stop, model=gmodel, usage=usage)


def make_client(project: Optional[str] = None) -> Any:
    """The client every AI step should use: the Gemini shim by default;
    Anthropic (direct API if ANTHROPIC_API_KEY is set, else Claude on Vertex)
    when MERIDIAN_LLM_PROVIDER=anthropic."""
    if provider() == "gemini":
        return GeminiClaudeShim(project=project)
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_USE_DIRECT_API"):
        from anthropic import Anthropic
        return Anthropic()
    from anthropic import AnthropicVertex
    return AnthropicVertex(project_id=project or os.environ["ANTHROPIC_VERTEX_PROJECT_ID"],
                           region=os.environ.get("CLOUD_ML_REGION", "global"))


if __name__ == "__main__":
    class FakeModels:
        def __init__(self, response):
            self.response, self.calls = response, []
        def generate_content(self, **kw):
            self.calls.append(kw)
            return self.response

    def fake_response(parts, finish="STOP", block=None, usage=(120, 340, 60)):
        return NS(candidates=[NS(finish_reason=NS(name=finish), content=NS(parts=parts))],
                  prompt_feedback=NS(block_reason=block),
                  usage_metadata=NS(prompt_token_count=usage[0], thoughts_token_count=usage[1],
                                    candidates_token_count=usage[2]))

    def shim(response):
        fm = FakeModels(response)
        return GeminiClaudeShim(project="p", genai_client=NS(models=fm)), fm

    TOOL = {"name": "decide", "description": "d",
            "input_schema": {"type": "object", "properties": {"go": {"type": "boolean"}}, "required": ["go"]}}

    print("TEST 1: forced tool call -> Gemini function calling (mode ANY, that name), thinking on, Claude-shaped reply")
    s, fm = shim(fake_response([NS(thought=True, text="private reasoning", function_call=None),
                                NS(thought=False, text=None, function_call=NS(name="decide", args={"go": True}))]))
    r = s.messages.create(model="claude-opus-5", max_tokens=900, tools=[TOOL],
                          tool_choice={"type": "tool", "name": "decide"},
                          messages=[{"role": "user", "content": "should we?"}])
    call = fm.calls[0]
    cfg = call["config"]
    assert call["model"] == DEFAULT_STRONG
    assert cfg.tool_config.function_calling_config.mode == "ANY"
    assert cfg.tool_config.function_calling_config.allowed_function_names == ["decide"]
    assert cfg.tools[0].function_declarations[0].name == "decide"
    assert cfg.thinking_config.thinking_budget == 2048 and cfg.max_output_tokens == 900 + 2048
    assert call["contents"][0].parts[0].text == "should we?"
    assert r.stop_reason == "tool_use" and [b.type for b in r.content] == ["tool_use"]
    assert r.content[0].name == "decide" and r.content[0].input == {"go": True}, "thought part never surfaces"
    assert r.usage.thinking_tokens == 340 and r.model == DEFAULT_STRONG
    print("✓ opus -> strong model, budget 2048, thought hidden, tool_use block returned")

    print("TEST 1b: an oversized request is capped at the model's output limit instead of a 400")
    s, fm = shim(fake_response([NS(thought=False, text="ok", function_call=None)]))
    s.messages.create(model="claude-haiku-4-5", max_tokens=68000, messages=[{"role": "user", "content": "x"}])
    assert fm.calls[0]["config"].max_output_tokens == MAX_OUTPUT_TOKENS - 1
    print("✓ 68000 + thinking -> capped at 65535")

    print("TEST 1c: concurrent first use builds exactly ONE underlying client")
    import sys, time as _t
    from concurrent.futures import ThreadPoolExecutor
    built = []
    class SlowClient:
        def __init__(self, **kw):
            _t.sleep(0.05); built.append(self); self.models = FakeModels(fake_response([]))
    fake_genai = NS(Client=SlowClient)
    real = sys.modules.get("google.genai")
    sys.modules["google"].genai = sys.modules["google.genai"] = fake_genai     # what `from google import genai` resolves
    try:
        lazy = GeminiClaudeShim(project="p")
        with ThreadPoolExecutor(8) as pool:
            clients = list(pool.map(lambda _: lazy._client(), range(8)))
    finally:
        sys.modules["google.genai"] = real
        sys.modules["google"].genai = real
    assert len(built) == 1 and all(c is built[0] for c in clients), len(built)
    print("✓ 8 threads -> 1 client")

    print("TEST 2: tier mapping and overrides")
    assert gemini_model_for("claude-haiku-4-5@20251001") == DEFAULT_FAST
    assert gemini_model_for("claude-sonnet-5") == DEFAULT_STRONG
    assert gemini_model_for("gemini-2.5-pro") == "gemini-2.5-pro"
    assert thinking_budget_for(DEFAULT_FAST) == 1024
    os.environ["MERIDIAN_GEMINI_STRONG"] = "gemini-2.5-pro"
    os.environ["MERIDIAN_GEMINI_THINKING_STRONG"] = "512"
    assert gemini_model_for("claude-opus-5") == "gemini-2.5-pro" and thinking_budget_for("gemini-2.5-pro") == 512
    del os.environ["MERIDIAN_GEMINI_STRONG"], os.environ["MERIDIAN_GEMINI_THINKING_STRONG"]
    print("✓ ok")

    print("TEST 3: plain chat (system + text) -> text block; no tools config")
    s, fm = shim(fake_response([NS(thought=False, text="Answer.", function_call=None)]))
    r = s.messages.create(model="claude-sonnet-5", max_tokens=500, system="be brief",
                          messages=[{"role": "user", "content": "q"}])
    assert fm.calls[0]["config"].system_instruction == "be brief" and fm.calls[0]["config"].tools is None
    assert r.stop_reason == "end_turn" and r.content[0].type == "text" and r.content[0].text == "Answer."
    print("✓ ok")

    print("TEST 4: safety blocks -> 'refusal'; truncation -> 'max_tokens'; empty -> no blocks")
    for resp, want in ((fake_response([], finish="SAFETY"), "refusal"),
                       (fake_response([], block="PROHIBITED_CONTENT"), "refusal"),
                       (fake_response([NS(thought=False, text="partial", function_call=None)], finish="MAX_TOKENS"), "max_tokens"),
                       (NS(candidates=[], prompt_feedback=None, usage_metadata=None), "end_turn")):
        s, _ = shim(resp)
        assert s.messages.create(model="claude-opus-5", max_tokens=10, messages=[{"role": "user", "content": "x"}]).stop_reason == want
    print("✓ ok")

    print("TEST 5: provider switch")
    os.environ.pop("MERIDIAN_LLM_PROVIDER", None)
    assert provider() == "gemini" and isinstance(make_client("p"), GeminiClaudeShim)
    os.environ["MERIDIAN_LLM_PROVIDER"] = "nonsense"
    assert provider() == "gemini", "an unknown value falls back to the default, not to a crash"
    os.environ["MERIDIAN_LLM_PROVIDER"] = "anthropic"
    assert provider() == "anthropic"
    del os.environ["MERIDIAN_LLM_PROVIDER"]
    print("✓ ok")

    print("TEST 6: the existing callers' parsing works unchanged on the shim's output")
    import proactive_recal as pr
    s, _ = shim(fake_response([NS(thought=False, text=None, function_call=NS(
        name="decide_recalibration", args={"recalibrate": True, "confidence": 0.8, "urgency": "high", "reason": "x"}))]))
    assert pr.call_claude(s, "prompt")["recalibrate"] is True
    s, _ = shim(fake_response([], finish="SAFETY"))
    try:
        pr.call_claude(s, "prompt")
        raise AssertionError("refusal must raise so the caller falls back")
    except ValueError as e:
        assert "refusal" in str(e)
    print("✓ proactive_recal.call_claude parses a Gemini reply; a Gemini safety block triggers its fallback")
    print("\nAll llm tests passed.")

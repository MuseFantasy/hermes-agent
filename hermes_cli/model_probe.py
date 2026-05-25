"""Model capability probe — auto-detect context window from provider /v1/models.

Use case: routing aliases (e.g. ``ark-code-latest`` on Volcengine Ark Coding
Plan, ``openrouter/auto``) where the configured model name does not directly
map to a known context window. Probes the provider's ``/v1/models`` endpoint,
finds the currently routed model, extracts ``token_limits.context_window``
(or equivalent), and persists it to ``~/.hermes/config.yaml`` under
``model.context_length``.

Triggered by ``hermes model --probe``. No automatic execution on session
start — explicit user action only (consistent with user preference to avoid
silent API calls).
"""

from __future__ import annotations

import json
import sys
from typing import Optional, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


# Routing aliases whose real context window can only be determined by probing
# the provider endpoint. Add new aliases here as they appear.
_ROUTING_ALIASES = {
    "ark-code-latest",
    "ark-code",
    "openrouter/auto",
    "auto",
}


def _read_model_config() -> Tuple[str, str, str]:
    """Return (model, base_url, api_key) from current Hermes config.

    Reads via the same path resolution as the CLI uses to avoid drift.
    """
    from hermes_cli.config import load_config  # type: ignore

    cfg = load_config()
    model_section = cfg.get("model", {}) or {}
    model = (model_section.get("default") or model_section.get("name") or "").strip()
    base_url = (model_section.get("base_url") or "").strip().rstrip("/")
    api_key = (model_section.get("api_key") or "").strip()
    return model, base_url, api_key


def _fetch_models_catalog(base_url: str, api_key: str, timeout: float = 10.0) -> dict:
    """GET {base_url}/models — returns the parsed JSON body.

    Most OpenAI-compatible providers expose this endpoint and include
    per-model metadata (context_window, max_output_tokens, etc.) when
    the upstream actually publishes those limits.
    """
    url = f"{base_url}/models"
    req = urlrequest.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _extract_context_window(model_entry: dict) -> Optional[int]:
    """Try several common key paths to pull the context window out of an entry.

    Different providers nest this metadata differently:
      - Volcengine Ark: token_limits.context_window
      - OpenRouter:     context_length (top-level)
      - OpenAI:         not exposed in /v1/models
      - Anthropic:      not exposed in /v1/models
    """
    # Volcengine Ark style
    token_limits = model_entry.get("token_limits") or {}
    if isinstance(token_limits, dict):
        for k in ("context_window", "max_context", "context_length"):
            v = token_limits.get(k)
            if isinstance(v, int) and v > 0:
                return v

    # Top-level keys
    for k in ("context_length", "context_window", "max_context_length", "max_tokens"):
        v = model_entry.get(k)
        if isinstance(v, int) and v > 0:
            return v

    return None


def _ping_routed_model(
    base_url: str, api_key: str, model: str, timeout: float = 30.0
) -> Optional[str]:
    """Send a minimal chat completion to learn which underlying model the
    routing alias resolves to. Returns the ``model`` field from the response.

    Costs a few tokens of API credit. Only invoked when explicit probing
    can't resolve the alias via /v1/models alone.
    """
    url = f"{base_url}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        routed = data.get("model")
        if isinstance(routed, str) and routed.strip():
            return routed.strip()
    except (HTTPError, URLError, json.JSONDecodeError, KeyError):
        return None
    return None


def _write_context_length(value: int) -> Tuple[Optional[int], int]:
    """Persist model.context_length to ~/.hermes/config.yaml.

    Returns ``(old_value, new_value)``. ``old_value`` is None when the key
    was not previously set.
    """
    from hermes_cli.config import load_config, save_config  # type: ignore

    cfg = load_config()
    model_section = cfg.setdefault("model", {})
    old = model_section.get("context_length")
    model_section["context_length"] = int(value)
    save_config(cfg)
    return (int(old) if isinstance(old, int) else None, int(value))


def probe_and_apply(verbose: bool = True) -> int:
    """Run the full probe → write flow. Returns POSIX exit code."""
    try:
        model, base_url, api_key = _read_model_config()
    except Exception as exc:
        print(f"[ERR] Failed to read Hermes config: {exc}", file=sys.stderr)
        return 2

    if not model or not base_url or not api_key:
        print(
            "[ERR] model.default / model.base_url / model.api_key are required "
            "in ~/.hermes/config.yaml to run probe.",
            file=sys.stderr,
        )
        return 2

    if verbose:
        print(f"[INFO] Probing provider: {base_url}")
        print(f"[INFO] Configured model: {model}")

    # 1. Fetch /v1/models
    try:
        catalog = _fetch_models_catalog(base_url, api_key)
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        print(f"[ERR] GET {base_url}/models failed: {exc}", file=sys.stderr)
        return 3

    entries = catalog.get("data") or catalog.get("models") or []
    if not isinstance(entries, list) or not entries:
        print(f"[ERR] {base_url}/models returned no model entries", file=sys.stderr)
        return 3

    if verbose:
        print(f"[INFO] Catalog returned {len(entries)} models")

    # 2. Find the matching entry
    target_model = model
    ctx_window: Optional[int] = None

    # Direct match first
    direct = next(
        (e for e in entries if isinstance(e, dict) and e.get("id") == model), None
    )
    if direct:
        ctx_window = _extract_context_window(direct)

    # If alias and no direct match (or no ctx in direct), ping to resolve
    if ctx_window is None and model in _ROUTING_ALIASES:
        if verbose:
            print(
                f"[INFO] {model} is a routing alias — pinging to discover "
                "underlying model"
            )
        routed = _ping_routed_model(base_url, api_key, model)
        if routed:
            target_model = routed
            if verbose:
                print(f"[INFO] Routed to: {routed}")
            # Find routed model in catalog (fuzzy: prefix match handles
            # version suffixes like deepseek-v4-pro-260425)
            for e in entries:
                if not isinstance(e, dict):
                    continue
                eid = e.get("id", "")
                if eid == routed or eid.startswith(routed):
                    ctx_window = _extract_context_window(e)
                    if ctx_window:
                        target_model = eid
                        break

    if ctx_window is None or ctx_window <= 0:
        print(
            f"[ERR] Could not extract context_window for model={target_model!r} "
            f"from {base_url}/models. Check provider metadata format.",
            file=sys.stderr,
        )
        return 4

    # 3. Persist
    try:
        old, new = _write_context_length(ctx_window)
    except Exception as exc:
        print(f"[ERR] Failed to write config: {exc}", file=sys.stderr)
        return 5

    print()
    print("Probe result")
    print(f"  Resolved model       : {target_model}")
    print(f"  Detected context     : {new:,} tokens")
    if old is None:
        print("  Previous setting     : (unset — was using fallback)")
    elif old == new:
        print(f"  Previous setting     : {old:,} (no change needed)")
    else:
        print(f"  Previous setting     : {old:,}")
        print(f"  Updated config.yaml  : model.context_length = {new}")
    print()
    print("Next sessions will use the new value. Restart current session to apply.")
    return 0


if __name__ == "__main__":  # manual invocation aid
    sys.exit(probe_and_apply())

"""
Course broker client — OpenAI-compatible endpoint.

Exports:
  call_model(system, user, model, temperature, max_tokens) -> (text, usage)
  extract_json(text) -> Any
"""

import json
import os
import re
import threading
import time
import logging
from typing import Any, Optional, Tuple

from openai import OpenAI

from .config import (
    API_BASE_URL, API_KEY,
    MAIN_MODEL, VALID_MODELS,
    TEMPERATURE_FACTUAL, TOP_P, SEED,
    MAX_RETRIES, RETRY_DELAY,
)

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        key = API_KEY or os.environ.get("COURSE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "API key not set. Run: import os; os.environ['COURSE_API_KEY'] = 'your-key'"
            )
        if not API_BASE_URL:
            raise RuntimeError("Set COURSE_API_BASE_URL before running Live mode.")
        _thread_local.client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=300.0)
    return _thread_local.client


# ~100K tokens at 4 chars/token — well under both 128K and 256K hard limits.
# The trim targets the user message (evidence) not the system prompt.
_MAX_USER_CHARS = 400_000


def call_model(
    system_prompt: str,
    user_prompt:   str,
    model:         str   = None,
    temperature:   float = None,
    max_tokens:    int   = None,
) -> Tuple[str, dict]:
    model_name  = model       or MAIN_MODEL
    temp        = temperature if temperature is not None else TEMPERATURE_FACTUAL
    num_predict = max_tokens  or 1024

    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model_name}'. Must be one of: {sorted(VALID_MODELS)}")

    # Hard-trim the user message to stay under the server's context-window limit.
    # Evidence sections are at the end of the user message, so trimming from the
    # tail discards the least-important evidence last.
    if len(user_prompt) > _MAX_USER_CHARS:
        logger.warning(
            "call_model: user_prompt truncated %d → %d chars to avoid hard limit",
            len(user_prompt), _MAX_USER_CHARS,
        )
        user_prompt = user_prompt[:_MAX_USER_CHARS] + "\n[... input truncated ...]"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_client().chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temp,
                max_tokens=num_predict,
            )
            text   = resp.choices[0].message.content
            finish = resp.choices[0].finish_reason
            usage  = {
                "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else -1,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else -1,
            }
            if finish == "length":
                logger.warning(
                    "finish_reason=length (token limit hit): in=%d out=%d max_tokens=%d",
                    usage["prompt_tokens"], usage["completion_tokens"], num_predict,
                )
            else:
                logger.debug(
                    "finish_reason=%s in=%d out=%d",
                    finish, usage["prompt_tokens"], usage["completion_tokens"],
                )
            return text, usage
        except Exception as exc:
            last_exc = exc
            logger.warning("call_model attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def extract_json(text: str) -> Any:
    """
    Return the first valid JSON object found in *text*.
    Strips markdown fences, fixes literal newlines, repairs truncated JSON.
    Always returns a dict.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*",           "", cleaned).strip()
    cleaned = _fix_literal_newlines(cleaned)

    try:
        return _ensure_dict(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        for m in reversed(list(re.finditer(pattern, cleaned))):
            try:
                return _ensure_dict(json.loads(m.group()))
            except json.JSONDecodeError:
                continue

    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        return _ensure_dict(repaired)

    # Last resort: scan for any complete JSON objects inside known array keys.
    # Handles cases where the outer structure is too broken for bracket-balancing repair
    # (e.g., truncated mid-key-name at the very first entity's attributes).
    partial = _extract_partial_arrays(cleaned)
    if partial:
        logger.warning("extract_json: partial extraction recovered %s",
                       {k: len(v) for k, v in partial.items()})
        return partial

    raise ValueError(f"No valid JSON found in model output:\n{text[:300]}")


def _ensure_dict(value: Any) -> dict:
    if isinstance(value, list):
        return {"items": value}
    return value


def _fix_literal_newlines(text: str) -> str:
    result, in_string, escape_next = [], False, False
    for ch in text:
        if escape_next:
            escape_next = False; result.append(ch); continue
        if ch == '\\' and in_string:
            escape_next = True; result.append(ch); continue
        if ch == '"':
            in_string = not in_string; result.append(ch); continue
        if in_string:
            if   ch == '\n': result.append('\\n')
            elif ch == '\r': result.append('\\r')
            elif ch == '\t': result.append('\\t')
            else:            result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def _extract_partial_arrays(text: str) -> Optional[dict]:
    """
    Scan text for complete JSON objects inside known distiller array keys.

    Used when _repair_truncated_json cannot reconstruct the outer structure —
    for example when the model was truncated mid-key-name inside the very first
    entity's attributes and the bracket-balancing approach produces an invalid
    key-without-value fragment.

    Returns a partial dict (e.g. {"entities": [obj1], "facts": []}) with
    whatever complete objects could be parsed, or None if nothing was found.
    """
    _ARRAY_KEYS = ("entities", "facts", "events", "claims", "themes")
    result: dict = {}

    for key in _ARRAY_KEYS:
        m = re.search(r'"' + key + r'"\s*:\s*\[', text)
        if not m:
            continue

        items: list = []
        i           = m.end()          # position right after '['
        depth       = 0
        in_str      = False
        esc_next    = False
        obj_start   = None

        while i < len(text):
            ch = text[i]
            if esc_next:
                esc_next = False
            elif ch == '\\' and in_str:
                esc_next = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        try:
                            items.append(json.loads(text[obj_start:i + 1]))
                        except json.JSONDecodeError:
                            pass
                        obj_start = None
                elif ch == ']' and depth == 0:
                    break   # clean end of array
            i += 1

        if items:
            result[key] = items

    return result if result else None


def _repair_truncated_json(text: str) -> Any:
    start = text.find('{')
    if start == -1:
        return None
    fragment    = text[start:]
    stack       = []
    in_string   = False
    escape_next = False
    for ch in fragment:
        if escape_next:
            escape_next = False; continue
        if ch == '\\' and in_string:
            escape_next = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    trimmed = fragment.rstrip()
    if in_string:
        trimmed += '"'
    if trimmed.endswith(','):
        trimmed = trimmed[:-1]
    closing = ''.join('}' if c == '{' else ']' for c in reversed(stack))

    # Try repair, then progressively strip trailing incomplete elements on failure.
    # Handles cases like "key": truncated mid-key-name → "key" has no value → invalid.
    # Each iteration strips the content after the last comma and retries.
    attempt = trimmed
    for _ in range(6):
        candidate = re.sub(r',\s*([}\]])', r'\1', attempt + closing)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            idx = attempt.rfind(',')
            if idx <= 0:
                break
            attempt = attempt[:idx].rstrip()
    return None

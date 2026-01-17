"""JSON utilities.

We frequently ask LLMs to return JSON. In practice, providers sometimes wrap
JSON in markdown fences, prepend commentary, or return empty strings on errors.
This module provides a safe extractor to pull the first JSON object found.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_first_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract the first JSON object from a string.

    Returns (obj, error). If obj is None, error is a short message.
    """
    if not text or not isinstance(text, str):
        return None, "empty_response"

    # 1) Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # 2) Markdown fenced block
    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj, None
        except Exception:
            pass

    # 3) Best-effort: find a JSON object by braces (non-greedy)
    # This is intentionally conservative: we take the first balanced-looking object.
    brace_start = text.find("{")
    if brace_start == -1:
        return None, "no_json_object_found"

    # Scan for a matching closing brace.
    depth = 0
    for i in range(brace_start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj, None
                except Exception:
                    break

    return None, "json_parse_failed"

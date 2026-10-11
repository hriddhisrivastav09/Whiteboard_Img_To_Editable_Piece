"""
Local Ollama LLM cleanup.

Two calls, both OPTIONAL and safe to fail:
  1) Label normalization (fix OCR noise, keep ids stable).
  2) Summary + observations.

Never used for geometry / node detection / edge inference.
Any failure -> pipeline continues with classical results.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import requests

from .prompts import build_label_cleanup_prompt, build_summary_prompt


def _ollama_generate(host: str, model: str, prompt: str, timeout: int = 120) -> Optional[str]:
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json().get("response", "")
    except Exception:
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction that handles code fences, trailing commas,
    stray escapes, and leading/trailing prose — mirrors the approach in
    vlm_extractor._safe_json_parse for consistency."""
    if not text:
        return None

    # 1) Direct parse (fast path for well-behaved output)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3) Fix trailing commas before closing braces/brackets
    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    # 4) Fix stray backslash escapes
    def _fix_escapes(s: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    for transform in (_remove_trailing_commas, _fix_escapes,
                      lambda s: _remove_trailing_commas(_fix_escapes(s))):
        try:
            parsed = json.loads(transform(cleaned))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 5) Find the largest balanced {...} block in whatever the model returned
    best: Optional[Dict[str, Any]] = None
    best_len = 0
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(cleaned):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    block = cleaned[start:i + 1]
                    if len(block) > best_len:
                        for t in (lambda s: s, _remove_trailing_commas,
                                  _fix_escapes,
                                  lambda s: _remove_trailing_commas(_fix_escapes(s))):
                            try:
                                p = json.loads(t(block))
                                if isinstance(p, dict):
                                    best = p
                                    best_len = len(block)
                                    break
                            except Exception:
                                pass
    return best


def clean_labels(host: str, model: str, nodes: List[Dict[str, Any]], timeout: int = 120) -> List[Dict[str, Any]]:
    if not nodes:
        return nodes
    prompt = build_label_cleanup_prompt(nodes)
    raw = _ollama_generate(host, model, prompt, timeout=timeout)
    data = _extract_json(raw or "")
    if not data or "nodes" not in data:
        return nodes
    lookup = {item.get("id"): item.get("label") for item in data["nodes"] if isinstance(item, dict)}
    changed = 0
    for n in nodes:
        new_lbl = lookup.get(n["id"])
        if isinstance(new_lbl, str) and new_lbl.strip() and new_lbl.strip() != n["label"]:
            n["original_label"] = n["label"]
            n["label"] = new_lbl.strip()
            changed += 1
    if changed:
        # small confidence bump for successfully-cleaned labels
        for n in nodes:
            if "original_label" in n:
                n["confidence"] = round(min(0.99, n.get("confidence", 0.5) + 0.05), 3)
    return nodes


def summarize(
    host: str,
    model: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    stats: Dict[str, Any],
    timeout: int = 120,
    fallback_summary: str = "",
) -> Dict[str, Any]:
    """Generate an LLM summary.  If the LLM is unreachable or returns
    unparseable JSON, fall back to *fallback_summary* (a pre-built rule-based
    string passed in from pipeline.py) so the UI never shows an empty summary.
    """
    prompt = build_summary_prompt(nodes, edges, stats)
    raw = _ollama_generate(host, model, prompt, timeout=timeout)
    data = _extract_json(raw or "")
    if not data:
        return {
            "summary": fallback_summary,
            "observations": [],
        }
    return {
        "summary": (data.get("summary") or fallback_summary).strip(),
        "observations": [o for o in (data.get("observations") or []) if isinstance(o, str)],
    }


def ollama_available(host: str, timeout: int = 4) -> bool:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

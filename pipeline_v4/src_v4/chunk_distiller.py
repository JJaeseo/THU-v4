"""
Phase 2 — Chunk Distiller.

One LLM call per chunk produces structured JSON:
{
  "entities": [{"name":"...", "type":"...", "attributes":{"area_km2":1234.5,"country":"..."}}],
  "facts":    [{"subject":"...", "predicate":"...", "value":"...", "unit":"...", "normalized_value":1234.5}],
  "events":   [{"year_text":"1967", "year_approx":1967, "description":"...", "entity":"..."}],
  "claims":   [{"text":"...", "entity":"..."}],
  "themes":   [{"topic":"...", "entity":"...", "context":"..."}]
}

COMPLETENESS MANDATE: uncertain → include. Missing a relevant record is a critical failure.
Results cached to CHUNK_DISTILLS_DIR/{chunk_id}_{content_hash}.json.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Tuple

from .mistral_client import call_model, extract_json
from .config import (
    DISTILLER_MODEL, MAX_TOKENS_DISTILLER,
    TEMPERATURE_FACTUAL,
    CHUNK_DISTILLS_DIR,
)

logger = logging.getLogger(__name__)

_SYSTEM_BASE = """\
You are a structured knowledge extractor for a long-document QA pipeline.
Extract ALL relevant information from [CORE CONTENT] into structured JSON.

COMPLETENESS MANDATE:
- When uncertain whether to include a record — ALWAYS INCLUDE IT.
- Missing a relevant entity, fact, or theme is a critical failure.
- Only skip content that is clearly irrelevant (navigation text, page numbers, decorative text).

Output ONLY valid JSON — no other text:
{
  "entities": [
    {"name": "Entity Name", "type": "entity_type", "attributes": {"location": "Region A", "area_km2": 1234.5, "elevation_m": 2500, "founded_year": 1950, "count": 42}}
  ],
  "facts": [
    {"subject": "Entity Name", "predicate": "area_km2", "value": "1234.5", "unit": "km2", "normalized_value": 1234.5}
  ],
  "events": [
    {"year_text": "1967", "year_approx": 1967, "description": "Organisation formally established", "entity": "Entity Name"}
  ],
  "claims": [
    {"text": "Entity Name is the largest of its kind in the region", "entity": "Entity Name"}
  ],
  "themes": [
    {"topic": "industrialisation", "entity": "Entity Name", "context": "central to the region's economic transformation"}
  ]
}

Rules:
- Extract from [CORE CONTENT] only. Use [PREVIOUS CONTEXT] and [NEXT CONTEXT] for boundary awareness only.
- normalized_value: numeric float of the extracted quantity in CANONICAL UNIT.
  Area → km²  (1 ha = 0.01 km²,  1 acre = 0.004047 km²,  1 mile² = 2.589 km²)
  Elevation → metres  (1 foot = 0.3048 m)
  Distance → km
  null if value is not a pure number.
- attributes: include ALL categorical and numeric properties (country, region, size, capacity, population, founding year, dates, identifiers, and any other stated property)
- year_approx: integer year; negative for BC (e.g. -500 for 500 BC). null if unknown.
- context in themes: ≤ 120 characters. Summarise — do NOT copy verbatim.
- Do NOT invent information not in the text.

WHEN THE TEXT CONTAINS A DATA TABLE:
Extract every data row as individual facts — one fact per (row × numeric column) cell:
  subject          = the row label (leftmost column: country name, entity, category, item)
  predicate        = column header converted to snake_case
                     (e.g., "employee_observations", "response_rate_pct",
                     "weighting_efficiency_pct", "employer_count")
  value            = exact cell value as a string (e.g., "2 474", "29%", "408", "44%")
  unit             = column unit; if the table has an identifier visible in the text
                     (e.g. "Table 5.1", "Annex Table 5.A.3"), append it in brackets:
                     "observations [Table 5.1]"   or   "% [Annex Table 5.A.3]"
                     If no table name is visible in this chunk, use the unit alone or "".
  normalized_value = numeric float (strip "%" and whitespace: "2 474"→2474.0, "29%"→29.0),
                     null if the cell is non-numeric or a label
⚠ Extract EVERY ROW individually — summarising or skipping rows is a CRITICAL FAILURE.
  Missed rows cause wrong superlative answers and undetected cross-table contradictions.

WHEN THE TEXT CONTAINS RELEASE NOTE SECTIONS (version headers, build numbers, CVE IDs):

1. RELEASE NOTE SECTION HEADER — extract each section as an entity:
   type: "release_note_section"
   attributes: {
     "product": "ESXi" | "vCenter Server" | "VMware Host Client" | ...,
     "version": "7.0",
     "update_letter": "3w",          ← e.g. 3w, 3v, 3s, 3r …
     "release_date": "DD MMM YYYY",  ← exact text from document
     "build_number": "12345678"      ← ISO Build or build number stated
   }
   Extract EVERY release note section that starts in [CORE CONTENT].
   Patterns to recognise: "VMware ESXi 7.0 Update 3w Release Notes",
   "ESXi 7.0 | 15 FEB 2024 | ISO Build 23299997".

2. CVE IDENTIFIERS — extract every CVE-YYYY-NNNNN occurrence:
   entity: {"name":"CVE-2021-22005","type":"cve_identifier",
            "attributes":{"year":2021,"cve_id":"CVE-2021-22005","product":"vCenter Server"}}
   claim:  {"text":"<full sentence describing the fix or vulnerability for this CVE>",
            "entity":"CVE-2021-22005"}
   ⚠ Extract EVERY CVE ID — used for counting, year-ranging, and contradiction detection.
   ⚠ Also record which release note section the CVE appeared in, as a fact:
   fact: {"subject":"CVE-2021-22005","predicate":"referenced_in_release",
          "value":"vCenter Server 7.0 Update 3w","unit":"","normalized_value":null}

3. VMSA SECURITY ADVISORIES — extract every VMSA-YYYY-XXXX occurrence:
   entity: {"name":"VMSA-2021-0020","type":"vmsa_advisory",
            "attributes":{"year":2021,"vmsa_id":"VMSA-2021-0020"}}
   fact:   {"subject":"VMSA-2021-0020","predicate":"referenced_in_release",
            "value":"ESXi 7.0 Update 3w","unit":"","normalized_value":null}

4. DOWNLOAD ARTIFACTS — extract size and checksums when stated:
   fact: {"subject":"ESXi 7.0 Update 3w depot","predicate":"download_size_mb",
          "value":"617.9","unit":"MB","normalized_value":617.9}
   fact: {"subject":"ESXi 7.0 Update 3w","predicate":"md5sum",
          "value":"a1b2c3d4…","unit":"","normalized_value":null}
   fact: {"subject":"ESXi 7.0 Update 3w","predicate":"sha256sum",
          "value":"e5f6…","unit":"","normalized_value":null}

5. FIX COUNTS — when resolved-issues lists are present, count bug fixes vs security fixes:
   fact: {"subject":"ESXi 7.0 Update 3w","predicate":"bug_fix_count",
          "value":"12","unit":"fixes","normalized_value":12.0}
   fact: {"subject":"ESXi 7.0 Update 3w","predicate":"security_fix_count",
          "value":"5","unit":"fixes","normalized_value":5.0}

"""


def distill_chunk(
    chunk:       Dict[str, Any],
    demand_plan: Dict[str, Any],
    use_cache:   bool = True,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Distil *chunk* into structured JSON with one LLM call.
    Returns (distilled_json, usage_dict).
    """
    chunk_id = chunk["chunk_id"]
    os.makedirs(CHUNK_DISTILLS_DIR, exist_ok=True)

    system = _build_system_prompt(demand_plan)
    user   = _build_user_prompt(chunk)

    content_sig = hashlib.md5(
        chunk.get("core_content", "")[:300].encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    prompt_sig = hashlib.md5(
        system.encode("utf-8", errors="replace")
    ).hexdigest()[:6]
    cache_path = os.path.join(CHUNK_DISTILLS_DIR, f"{chunk_id}_{content_sig}_{prompt_sig}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_distilled = cached.get("distilled", _empty())
        # Skip empty cache entries — a previous JSON parse failure was cached.
        # Re-extract so _extract_partial_arrays (and any other repairs) get a chance.
        has_data = any(cached_distilled.get(k) for k in ("entities", "facts", "events", "claims", "themes"))
        if has_data:
            logger.debug("Cache hit: %s (%s)", chunk_id, content_sig)
            return cached_distilled, cached.get("usage", {})
        logger.info("Empty cache for %s — re-extracting (previous run had JSON parse failure)", chunk_id)

    raw, usage = call_model(
        system, user,
        model=DISTILLER_MODEL,
        temperature=TEMPERATURE_FACTUAL,
        max_tokens=MAX_TOKENS_DISTILLER,
    )

    try:
        distilled = extract_json(raw)
    except ValueError as exc:
        logger.warning("JSON parse failed for %s: %s", chunk_id, exc)
        distilled = _empty()

    distilled = _normalise(distilled)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"chunk_id": chunk_id, "distilled": distilled, "usage": usage},
                  f, ensure_ascii=False, indent=2)

    n_e  = len(distilled.get("entities", []))
    n_f  = len(distilled.get("facts",    []))
    n_ev = len(distilled.get("events",   []))
    n_c  = len(distilled.get("claims",   []))
    n_t  = len(distilled.get("themes",   []))
    logger.info(
        "%s → e=%d f=%d ev=%d cl=%d th=%d  in=%d/out=%d",
        chunk_id, n_e, n_f, n_ev, n_c, n_t,
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
    )

    return distilled, usage


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_system_prompt(demand_plan: Dict[str, Any]) -> str:
    entity_types = demand_plan.get("entity_types",       [])
    predicates   = demand_plan.get("numeric_predicates", [])
    triggers     = demand_plan.get("event_triggers",     [])
    topics       = demand_plan.get("theme_topics",       [])

    focus = ""
    if entity_types:
        types_str = ", ".join(entity_types)
        focus += (
            f"\n\n══ ENTITY EXTRACTION — MANDATORY COMPLETENESS ══"
            f"\nTarget entity types: {types_str}"
            f"\nFor EVERY entity type listed above you MUST:"
            f"\n  1. Extract EVERY single occurrence in [CORE CONTENT] — no exceptions, no skipping."
            f"\n     Even if an entity appears only in a heading, a list item, or a single sentence."
            f"\n  2. Include ALL attributes present in the text for each entity:"
            f"\n     location/region, size/area, numeric measurements, dates, identifiers,"
            f"\n     classification/type, affiliations, designations, and any other stated property."
            f"\n  3. After your initial extraction, re-scan the text once to verify no instance was missed."
            f"\n  ⚠ MISSING ANY ENTITY OF THESE TYPES = CRITICAL FAILURE."
            f"\n    It will produce wrong counts and incorrect answers to user questions."
        )
    if predicates:
        focus += (
            f"\n\nTarget numeric predicates — capture exact value + unit, normalize to canonical unit:"
            f"\n  {', '.join(predicates)}"
        )
    if triggers:
        focus += (
            f"\n\nTarget event types — record year_approx (int, negative=BC) + description for each:"
            f"\n  {', '.join(triggers)}"
        )
    if topics:
        focus += (
            f"\n\nTarget theme topics — record ANY mention, even a single sentence:"
            f"\n  {', '.join(topics)}"
        )

    return _SYSTEM_BASE + focus


def _build_user_prompt(chunk: Dict[str, Any]) -> str:
    parts = []
    prev = chunk.get("previous_context", "").strip()
    if prev:
        parts.append(f"[PREVIOUS CONTEXT]\n{prev}")
    parts.append(f"[CORE CONTENT]\n{chunk.get('core_content', '').strip()}")
    nxt = chunk.get("next_context", "").strip()
    if nxt:
        parts.append(f"[NEXT CONTEXT]\n{nxt}")
    return "\n\n".join(parts)


def _empty() -> Dict[str, Any]:
    return {"entities": [], "facts": [], "events": [], "claims": [], "themes": []}


def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("entities", "facts", "events", "claims", "themes"):
        if not isinstance(raw.get(key), list):
            raw[key] = []
    return raw

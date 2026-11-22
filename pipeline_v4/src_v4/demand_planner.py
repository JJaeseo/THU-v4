"""
Phase 1 — Demand Planner.

One LLM call analyses all questions and produces a demand_plan that:
  1. Tells the distiller what entity types / predicates / topics to extract
  2. Gives each question a solver hint used by Phase 3 solvers

Output is cached; re-read on subsequent runs unless force=True.

demand_plan structure:
{
  "entity_types": ["national_park", "lake", ...],
  "numeric_predicates": ["area_km2", "elevation_m", ...],
  "event_triggers": ["establishment", "eruption", ...],
  "theme_topics": ["glaciation", "conservation", ...],
  "question_plans": {
    "d01": {"solver": "aggregation", "entity_type": "...", "filter_key": "...", "filter_value": "..."},
    "d07": {"solver": "superlative",  "predicate": "...", "direction": "max"},
    "d10": {"solver": "absence",      "topic_keywords": ["...", "..."]},
    "d14": {"solver": "contradiction","predicate": "..."},
    "d15": {"solver": "needle",       "keywords": ["...", "..."]},
    "d18": {"solver": "cross_section","entity_types": ["..."], "keywords": ["..."]},
    "d21": {"solver": "global_synthesis"}
  }
}
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from .mistral_client import call_model, extract_json
from .config import (
    PLANNER_MODEL, MAX_TOKENS_PLANNER,
    TEMPERATURE_FACTUAL,
    DEMAND_PLAN_FILE, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

_Q_TYPE_DESCRIPTIONS = {
    "aggregation":      "Count/list entities satisfying a criterion",
    "superlative":      "Find the max/min entity on a numeric dimension",
    "absence":          "Confirm something is NOT mentioned in the document",
    "contradiction":    "Find conflicting claims within the document",
    "cross_section":    "Link information from multiple distinct document sections",
    "global_synthesis": "Summarise or characterise the document as a whole",
    "needle":           "Retrieve a specific fact stated at one location",
}


def build_demand_plan(
    questions: List[Dict[str, Any]],
    force:     bool = False,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Analyse *questions* and return (demand_plan, usage_dict).
    Caches result to DEMAND_PLAN_FILE; returns cached version unless force=True.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force and os.path.exists(DEMAND_PLAN_FILE):
        logger.info("Loading cached demand plan from %s", DEMAND_PLAN_FILE)
        with open(DEMAND_PLAN_FILE, "r", encoding="utf-8") as f:
            plan = json.load(f)
        return plan, {}

    logger.info("Building demand plan for %d questions…", len(questions))
    plan, usage = _call_planner_llm(questions)

    with open(DEMAND_PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    logger.info("Demand plan saved → %s", DEMAND_PLAN_FILE)

    return plan, usage


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_planner_llm(questions: List[Dict[str, Any]]) -> Tuple[Dict, dict]:
    q_block = _format_questions(questions)

    system = (
        "You are a demand planner for a long-document QA pipeline.\n"
        "Given a list of questions, produce a demand_plan that guides structured extraction from every document chunk.\n\n"
        "The pipeline extracts five record types from each chunk:\n"
        "  entities : named objects with typed attributes (name, country, area, elevation …)\n"
        "  facts    : (subject, predicate, value, unit, normalized_value) numeric triples\n"
        "  events   : dated occurrences with entity and description\n"
        "  claims   : statements that may conflict across sections\n"
        "  themes   : topic/concept presence markers\n\n"
        "Your demand_plan drives:\n"
        "  1. What the extractor looks for in every chunk (entity_types, numeric_predicates, …)\n"
        "  2. How Phase 3 solvers retrieve and compute answers (question_plans)\n\n"
        "Solver types and their required question_plan fields:\n"
        '  aggregation    → {"solver":"aggregation","entity_type":"...","filter_key":"...","filter_value":"..."}\n'
        '                   (omit filter_key/filter_value if no filter applies)\n'
        '  superlative    → {"solver":"superlative","predicate":"...","predicate_aliases":["...","..."],"direction":"max"|"min"}\n'
        '                   (predicate_aliases: 2-5 alternative names the distiller might use for the same concept)\n'
        '  absence        → {"solver":"absence","topic_keywords":["...","..."],'
        '                    "keyword_variants":{"topic_phrase":["synonym1","synonym2","synonym3"]}}\n'
        '                   (topic_keywords: the exact option phrases from the question;\n'
        '                    keyword_variants: for EACH topic_keyword, list 3-6 alternative phrasings\n'
        '                    or related terms the document might use to discuss the same concept)\n'
        '  contradiction  → {"solver":"contradiction","predicate":"..."}\n'
        '  needle         → {"solver":"needle","keywords":["...","..."]}\n'
        '                   (specific search terms that would appear near the answer)\n'
        '  cross_section  → {"solver":"cross_section","entity_types":["..."],"keywords":["..."]}\n'
        '  global_synthesis→{"solver":"global_synthesis"}\n\n'
        "Output valid JSON only — no other text.\n"
        "The JSON schema is shown below with TWO EXAMPLES — one for general documents, one for technical release notes.\n"
        "Choose entity_types and predicates appropriate for the REAL questions you receive.\n\n"
        "EXAMPLE A — general / historical document:\n"
        "{\n"
        '  "entity_types": ["building", "person", "organisation", "event"],\n'
        '  "numeric_predicates": ["height_m", "founded_year", "capacity", "distance_km"],\n'
        '  "event_triggers": ["founding", "discovery", "publication", "collapse"],\n'
        '  "theme_topics": ["industrialisation", "migration", "conflict", "trade"],\n'
        '  "question_plans": {\n'
        '    "q01": {"solver": "aggregation", "entity_type": "building", "filter_key": "country", "filter_value": "Italy"},\n'
        '    "q02": {"solver": "superlative", "predicate": "height_m", "predicate_aliases": ["elevation_m", "altitude_m", "peak_height"], "direction": "max"},\n'
        '    "q03": {"solver": "absence", "topic_keywords": ["famine", "starvation", "food shortage"],\n'
        '            "keyword_variants": {"famine": ["food crisis", "food scarcity", "hunger epidemic"],\n'
        '                                 "starvation": ["extreme hunger", "malnutrition"],\n'
        '                                 "food shortage": ["food supply crisis", "food scarcity"]}},\n'
        '    "q04": {"solver": "contradiction", "predicate": "founded_year"},\n'
        '    "q05": {"solver": "needle", "keywords": ["treaty", "signed", "ratified", "agreement"]},\n'
        '    "q06": {"solver": "cross_section", "entity_types": ["person", "organisation"], "keywords": ["collaboration", "joint", "shared"]},\n'
        '    "q07": {"solver": "global_synthesis"}\n'
        "  }\n"
        "}\n\n"
        "EXAMPLE B — technical release notes document (software versions, CVEs, build numbers):\n"
        "{\n"
        '  "entity_types": ["release_note_section", "cve_identifier", "vmsa_advisory", "software_component", "download_artifact"],\n'
        '  "numeric_predicates": ["build_number", "download_size_mb", "cve_year", "release_year", "bug_fix_count", "security_fix_count"],\n'
        '  "event_triggers": ["release", "patch", "fix", "deprecation", "removal", "upgrade"],\n'
        '  "theme_topics": ["security_vulnerability", "bug_fix", "deprecation", "upgrade_path", "brand_change", "cve_remediation"],\n'
        '  "question_plans": {\n'
        '    "q01": {"solver": "aggregation", "entity_type": "release_note_section", "filter_key": "product", "filter_value": "vCenter Server"},\n'
        '    "q02": {"solver": "cross_section", "entity_types": ["release_note_section"], "keywords": ["Update 3w", "build_number", "release_date", "ESXi", "vCenter Server"]},\n'
        '    "q03": {"solver": "superlative", "predicate": "cve_year", "predicate_aliases": ["year", "cve_identifier_year"], "direction": "min"},\n'
        '    "q04": {"solver": "needle", "keywords": ["md5sum", "checksum", "download", "depot", "Update 3w"]},\n'
        '    "q05": {"solver": "absence", "topic_keywords": ["ransomware", "VMSA security advisories", "knowledge base", "severity rating"],\n'
        '            "keyword_variants": {"ransomware": ["ransom", "ransomware attack", "encryption attack"],\n'
        '                                 "VMSA security advisories": ["VMSA", "security advisory", "VMware Security Advisory"],\n'
        '                                 "knowledge base": ["KB article", "knowledge base article", "VMware KB"],\n'
        '                                 "severity rating": ["CVSS", "severity score", "critical severity"]}},\n'
        '    "q06": {"solver": "contradiction", "predicate": "cve_description"},\n'
        '    "q07": {"solver": "global_synthesis"}\n'
        "  }\n"
        "}"
    )

    user = (
        f"Questions ({len(questions)} total):\n\n{q_block}\n\n"
        "Produce the demand_plan. Rules:\n"
        "- Use snake_case for predicate names (e.g. height_m, founded_year, population).\n"
        "- Include synonyms in topic_keywords and keyword lists.\n"
        "- Every question id MUST appear as a key in question_plans.\n"
        "- Be specific — vague plans produce poor extraction."
    )

    raw, usage = call_model(
        system, user,
        model=PLANNER_MODEL,
        temperature=TEMPERATURE_FACTUAL,
        max_tokens=MAX_TOKENS_PLANNER,
    )

    try:
        plan = extract_json(raw)
    except ValueError as exc:
        logger.error("Demand plan parse failed: %s\nRaw:\n%s", exc, raw[:500])
        plan = _empty_plan()

    # Ensure every question ID is covered
    qps = plan.setdefault("question_plans", {})
    for q in questions:
        q_id  = q.get("id")
        q_cat = q.get("category", "needle")
        if q_id not in qps:
            logger.warning("question_plans missing %s — inserting fallback", q_id)
            qps[q_id] = {"solver": q_cat}

    return plan, usage


def _empty_plan() -> Dict[str, Any]:
    return {
        "entity_types":       [],
        "numeric_predicates": [],
        "event_triggers":     [],
        "theme_topics":       [],
        "question_plans":     {},
    }


def _format_questions(questions: List[Dict[str, Any]]) -> str:
    lines = []
    for q in questions:
        q_id  = q.get("id", "?")
        q_cat = q.get("category", "needle")
        q_txt = q.get("question", "")
        desc  = _Q_TYPE_DESCRIPTIONS.get(q_cat, q_cat)
        lines.append(f"[{q_id}] ({q_cat} — {desc})\n  Q: {q_txt}")
    return "\n\n".join(lines)

"""
Phase 3 — Solvers.

Routes each question to the appropriate solver based on q_cat and question_plan.

Solver hierarchy:
  1. Python-deterministic (aggregation count, superlative max/min, contradiction set-diff)
  2. LLM synthesis on structured evidence from KnowledgeStore
  3. Needle fallback: keyword_search_raw on original chunk text → LLM synthesis
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .mistral_client import call_model, extract_json
from .knowledge_store import KnowledgeStore
from .config import SOLVER_MODEL, MAX_TOKENS_SOLVER, TEMPERATURE_ANSWER

logger = logging.getLogger(__name__)

# Appended to every LLM system prompt that may otherwise produce markdown or
# internal evidence labels (ENTITY [...], FACT:, CLAIM [], etc.) in the answer.
_STYLE_RULE = (
    "\nStyle rules (MANDATORY):\n"
    "- Write in plain, natural prose using complete sentences.\n"
    "- Do NOT use markdown: no **bold**, no ## headers, no bullet lists (- item).\n"
    "- Do NOT expose internal evidence labels such as ENTITY [...], FACT:, CLAIM [], "
    "  RAW:, ALL-FACTS, NEAR-MATCH, or any other pipeline-internal prefix.\n"
    "- Explain findings the way a knowledgeable person would in conversation.\n"
    "- If listing multiple items, use natural language ('First, ... Second, ... Finally, ...' "
    "  or 'These are: A, B, and C.')."
)

# Appended to solvers where fabricating specific facts is most damaging
# (needle, aggregation, llm_synthesis fallback).
_GROUNDING_RULE = (
    "\nGrounding rules (MANDATORY):\n"
    "- Every specific entity name, numeric value, date, and location you state "
    "  MUST appear verbatim in the evidence lines provided above.\n"
    "- Do NOT use knowledge from your training data — treat the evidence as the "
    "  only source of truth. If a fact is not in the evidence, it does not exist.\n"
    "- If you are about to write a specific number or name that is not in the evidence, "
    "  write instead: 'the evidence does not specify this.'"
)


def solve_question(
    question:    Dict[str, Any],
    store:       KnowledgeStore,
    demand_plan: Dict[str, Any],
) -> Tuple[str, Dict[str, int]]:
    q_id  = question.get("id", "")
    q_cat = question.get("category", "needle")
    q_txt = question.get("question", "")
    plan  = demand_plan.get("question_plans", {}).get(q_id, {})

    dispatch = {
        "aggregation":      _solve_aggregation,
        "superlative":      _solve_superlative,
        "absence":          _solve_absence,
        "contradiction":    _solve_contradiction,
        "needle":           _solve_needle,
        "cross_section":    _solve_cross_section,
        "global_synthesis": _solve_global_synthesis,
    }
    fn = dispatch.get(q_cat, _solve_generic)

    # Multi-part plan (list of sub-plans): solve each separately and join
    if isinstance(plan, list):
        if len(plan) > 1:
            parts: List[str] = []
            total_usage: Dict[str, int] = {}
            for sub in plan:
                if not isinstance(sub, dict):
                    continue
                sub_ans, sub_use = fn(q_txt, sub, store)
                parts.append(sub_ans)
                for k, v in (sub_use or {}).items():
                    total_usage[k] = total_usage.get(k, 0) + (v or 0)
            return ("\n\n".join(parts) if parts else "No answer found.", total_usage)
        plan = plan[0] if plan else {}

    if not isinstance(plan, dict):
        plan = {}

    return fn(q_txt, plan, store)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _build_inventory_evidence(inventory: Dict[str, Any], q_txt: str) -> List[str]:
    """
    Format inventory data as structured evidence lines for the LLM.
    Only includes sections relevant to the question (figure/box/table).
    Returns [] when no relevant data exists (natural language documents).
    """
    q_lower = q_txt.lower()
    lines: List[str] = []

    def _fmt(key: str, label: str, elem: str) -> None:
        data = inventory.get(key, {})
        if not data:
            return
        lines.append(f"=== INVENTORY: {label} ===")
        total = 0
        per_ch: Dict[str, int] = {}
        for ch in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            lbls = data[ch]
            n = len(lbls)
            total += n
            per_ch[ch] = n
            preview = ", ".join(lbls[:6])
            suffix  = f", …+{n - 6}" if n > 6 else ""
            lines.append(f"  Chapter {ch}: {n} {elem}  [{preview}{suffix}]")
        lines.append(f"  TOTAL {label}: {total}")
        if per_ch:
            max_n   = max(per_ch.values())
            max_chs = sorted(ch for ch, n in per_ch.items() if n == max_n)
            ch_str  = "  ".join(f"Chapter {c}" for c in max_chs)
            lines.append(f"  CHAPTER(S) WITH MOST: {ch_str} ({max_n} {elem} each)")

    want_fig   = any(w in q_lower for w in ("figure", "figures"))
    want_box   = any(w in q_lower for w in ("box", "boxes"))
    want_table = any(w in q_lower for w in ("table", "tables"))

    if want_fig:
        _fmt("figures",      "MAIN FIGURES",           "figures")
    if want_box:
        _fmt("boxes",        "NUMBERED BOXES",          "boxes")
    if want_table:
        _fmt("main_tables",  "MAIN NUMBERED TABLES",    "tables")
        _fmt("annex_tables", "CHAPTER-ANNEX TABLES",    "annex tables")

    return lines


def _solve_aggregation(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    # ── Inventory Track (Proposal C) ─────────────────────────────────────────
    # When the question counts structural labels (figures/boxes/tables),
    # the Python scanner has already produced exact counts — use them directly.
    inventory = store.get_inventory()
    if inventory:
        q_lower = q_txt.lower()
        _cnt = any(kw in q_lower for kw in ("how many", "count", "numbered", "total", "list"))
        _elm = any(kw in q_lower for kw in ("figure", "figures", "box", "boxes", "table", "tables"))
        if _cnt and _elm:
            inv_lines = _build_inventory_evidence(inventory, q_txt)
            if inv_lines:
                system = (
                    "You are answering a structural-counting question about a document.\n"
                    "A Python regex scanner has produced EXACT counts of all labeled elements.\n"
                    "INVENTORY LINES ARE DEFINITIVE — do NOT re-count from prose.\n"
                    "Rules:\n"
                    "- Answer each sub-part of the question directly from the INVENTORY.\n"
                    "- Use TOTAL lines for grand totals.\n"
                    "- Use CHAPTER(S) WITH MOST lines to identify the leading chapter.\n"
                    "- List every chapter's count in numerical order.\n"
                    "- If a chapter has 0 of an element, state 0 for that chapter.\n"
                    "- Do NOT invent counts or round figures."
                    + _STYLE_RULE
                )
                user = (
                    f"Question: {q_txt}\n\n"
                    "Document Inventory (exact counts from Python scanner):\n"
                    + "\n".join(inv_lines)
                    + "\n\nAnswer each sub-part directly from the inventory above:"
                )
                raw, usage = call_model(
                    system, user,
                    model=SOLVER_MODEL, temperature=0.0, max_tokens=MAX_TOKENS_SOLVER,
                )
                return raw.strip(), usage

    entity_type  = str(plan.get("entity_type",  "") or "")
    filter_key   = str(plan.get("filter_key",   "") or "")
    predicate    = str(plan.get("predicate",    "") or "")

    # filter_value: LLM may emit int (e.g. 3000) or list (e.g. ["Croatia","Montenegro"])
    _fv_raw = plan.get("filter_value", "")
    _filter_values: List[str] = []
    if isinstance(_fv_raw, list):
        _filter_values = [str(fv or "") for fv in _fv_raw if fv]
        _fv_raw = _fv_raw[0] if _fv_raw else ""
    filter_value = str(_fv_raw or "")
    # When multiple filter values exist, use them combined as the search query
    # (single LLM call avoids self-contradictory concatenated sub-answers)
    _combined_filter = " OR ".join(_filter_values) if len(_filter_values) > 1 else filter_value

    # ── Track A: All entities of the requested type (no pre-filtering) ────────
    all_entities = store.entities_by_type(entity_type) if entity_type else []
    entity_lines = []
    _seen_entity_names: set = set()
    for e in all_entities[:300]:
        _ename = str(e.get("name", "") or "").strip().lower()
        if _ename and _ename in _seen_entity_names:
            continue  # skip duplicate — same station/entity extracted from multiple chunks
        if _ename:
            _seen_entity_names.add(_ename)
        attrs    = e.get("attributes") or {}
        attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:6])
        line     = f"ENTITY [{e.get('type','')}]: {e.get('name','')}"
        if attr_str:
            line += f"  ({attr_str})"
        entity_lines.append(line)

    # ── Track B: Numeric facts (critical for threshold questions like ≥3000m) ─
    # facts_by_predicate returns structured numeric data with normalized_value,
    # which the LLM can compare against thresholds reliably.
    fact_lines = []
    pred = predicate or filter_key
    if pred:
        facts = store.facts_by_predicate(pred)
        for f in facts[:150]:
            val_str = f"{f.get('value','')} {f.get('unit','')}".strip()
            nv      = f.get("normalized_value", "")
            line    = f"FACT: {f.get('subject','')} / {f.get('predicate','')} / {val_str}"
            if nv != "":
                line += f"  (normalized={nv})"
            fact_lines.append(line)

    # ── Track C: Keyword search — broadens coverage beyond extracted entities ─
    kw_query = _combined_filter if _combined_filter else q_txt
    kw_lines = store.keyword_search(kw_query, limit=40)
    # For multi-value filters, also search each value separately to avoid missing entities
    if len(_filter_values) > 1:
        _seen_kw = {line[:80] for line in kw_lines}
        for _fv in _filter_values[1:]:
            for _line in store.keyword_search(_fv, limit=20):
                if _line[:80] not in _seen_kw:
                    _seen_kw.add(_line[:80])
                    kw_lines.append(_line)

    # ── Track D: Sentence search — catches entities in raw text missed by Phase 2
    sent_lines = store.sentence_search(q_txt, limit=20)

    # ── Combine: entity_lines FIRST so they are never crowded out by facts ────
    # entity_lines are the primary counting signal for aggregation questions.
    seen, combined = set(), []
    for line in entity_lines + fact_lines + kw_lines + sent_lines:
        key = line[:80]
        if key not in seen:
            seen.add(key)
            combined.append(line)

    if not combined:
        raw_paras = store.keyword_search_raw(q_txt, limit=15)
        combined  = raw_paras
        seen.update(p[:80] for p in combined)

    # ── Track E: raw paragraph search — always on for aggregation ────────────
    raw_paras = store.keyword_search_raw(_combined_filter or entity_type or q_txt, limit=25)
    for p in raw_paras:
        key = p[:80]
        if key not in seen:
            seen.add(key)
            combined.append(p)

    n_entities = len(entity_lines)
    note = (
        f"\nNOTE: Only {n_entities} entities of type '{entity_type}' were captured in "
        "structured records. The full document almost certainly contains MORE — raw text "
        "paragraphs (unlabelled above) may name additional items not captured structurally. "
        "COUNT ALL QUALIFYING ITEMS from every source above, not just the ENTITY lines."
    ) if n_entities < 50 else ""

    _multi_filter_note = (
        f"- The question asks about items in ANY of these categories: "
        + ", ".join(f'"{fv}"' for fv in _filter_values)
        + ". Give a SINGLE unified answer covering all categories together.\n"
    ) if len(_filter_values) > 1 else ""

    system = (
        "You are answering an aggregation question about a document.\n"
        "Aggregation questions ask to count and/or list items satisfying a criterion.\n"
        "Rules:\n"
        + _multi_filter_note
        + "- Apply the question's criterion to ALL evidence provided.\n"
        "- State the exact count of qualifying items.\n"
        "- List every qualifying item with relevant context (country, region, value).\n"
        "- If the question has multiple sub-parts, answer EACH sub-part separately.\n"
        "- For numeric thresholds (e.g., '3000 metres or more'), check each value carefully "
        "  and include only items that meet the threshold.\n"
        "- Raw document sentences may name items not in structured records — include them too.\n"
        "- DEDUPLICATION: ENTITY lines are already deduplicated by name. If the same item "
        "  appears in both ENTITY lines and raw text passages, count it ONCE. "
        "  Do not double-count an item just because it appears in multiple evidence sources.\n"
        "- Use ONLY the provided evidence. Do not invent items."
        + _STYLE_RULE
        + _GROUNDING_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        f"Evidence from full document scan:\n"
        + "\n".join(combined[:200])
        + note
        + "\n\nAnswer completely, covering all sub-parts of the question:"
    )
    raw, usage = call_model(system, user, model=SOLVER_MODEL, temperature=0.1, max_tokens=MAX_TOKENS_SOLVER)
    return raw.strip(), usage


# ── Superlative ───────────────────────────────────────────────────────────────

def _solve_superlative(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    predicate = str(plan.get("predicate", "") or "")
    direction = str(plan.get("direction", "max") or "max")

    candidates = store.facts_by_predicate(predicate) if predicate else []
    candidates = [c for c in candidates if c.get("normalized_value") is not None]

    # predicate_aliases: broaden search to alternative predicate names the distiller may have used
    aliases = plan.get("predicate_aliases", [])
    if isinstance(aliases, list) and aliases:
        seen_subjs = {c.get("subject") for c in candidates}
        for alias in aliases:
            alias_str = str(alias or "").strip()
            if not alias_str or alias_str == predicate:
                continue
            extra = [
                f for f in store.facts_by_predicate(alias_str)
                if f.get("normalized_value") is not None
                and f.get("subject") not in seen_subjs
            ]
            candidates.extend(extra)
            seen_subjs.update(f.get("subject") for f in extra)

    if not candidates:
        kw_lines = store.keyword_search(q_txt, limit=30)
        if kw_lines:
            return _llm_synthesis(q_txt, "superlative", kw_lines)
        raw_paras = store.keyword_search_raw(q_txt, limit=10)
        if raw_paras:
            return _llm_synthesis(q_txt, "superlative", raw_paras, raw=True)
        return "This information is not found in the document.", {}

    # Normalize area units to km²
    if "area" in predicate.lower():
        candidates = [_normalize_area(c) for c in candidates]
        candidates = [c for c in candidates if c.get("normalized_value") is not None]

    # If too few results, supplement with facts from predicates sharing name words.
    # This handles Phase 1/Phase 2 predicate naming divergence without hardcoding.
    if len(candidates) < 8 and predicate:
        pred_words = set(predicate.replace("_", " ").lower().split()) - {"m", "km", "km2", "per"}
        all_preds = {f.get("predicate", "") for f in (store._facts or []) if f.get("predicate")}
        seen_subjs = {c.get("subject") for c in candidates}
        for alt_pred in sorted(all_preds):
            if alt_pred == predicate:
                continue
            alt_words = set(alt_pred.replace("_", " ").lower().split())
            if pred_words & alt_words:
                extra = [
                    f for f in store.facts_by_predicate(alt_pred)
                    if f.get("normalized_value") is not None
                    and f.get("subject") not in seen_subjs
                ]
                if "area" in alt_pred.lower():
                    extra = [_normalize_area(f) for f in extra]
                    extra = [f for f in extra if f.get("normalized_value") is not None]
                candidates.extend(extra)
                seen_subjs.update(f.get("subject") for f in extra)

    # Fix B: when structured data is still sparse, supplement with raw text so the
    # LLM can extract values the Phase 2 distiller missed.
    # Always run raw supplement — gives LLM a chance to correct predicate-mismatch errors
    raw_supplement = store.sentence_search(q_txt, limit=30)
    if not raw_supplement:
        raw_supplement = store.keyword_search_raw(q_txt, limit=15)

    reverse = (direction == "max")
    ranked  = sorted(candidates, key=lambda c: float(c["normalized_value"]), reverse=reverse)

    evidence_lines = []
    for c in ranked[:15]:
        val_str = f"{c.get('value','')} {c.get('unit','')}".strip()
        nv = c.get("normalized_value", "")
        line = f"RANKED: {c.get('subject','')} / {c.get('predicate','')} / {val_str}"
        if nv not in ("", None):
            line += f"  (norm={nv})"
        evidence_lines.append(line)

    has_raw = bool(raw_supplement)
    system = (
        "You are answering a superlative question (highest/lowest/largest/smallest).\n"
        "Rules:\n"
        "- RANKED candidates are pre-sorted best-to-worst for the criterion; "
        "  normalized values are in a canonical unit (km² for area, metres for elevation).\n"
        "- If RANKED candidates appear complete, the first entry is the answer.\n"
        + (
            "- SUPPLEMENTARY RAW TEXT passages may name entities with larger/higher "
            "  values not yet captured in the structured ranking.\n"
            "  If a raw passage names an entity whose value beats the top ranked entry, "
            "  that entity is the true answer — say so explicitly.\n"
            if has_raw else ""
        )
        + "- State the EXACT NAME of the winning entity (e.g., the country name, park name, "
        "  person name) as it appears in the evidence. Do NOT refer to it vaguely as "
        "  'the one associated with' a table or annex — name the entity directly.\n"
        "- Include entity name, its value with unit, and any location context.\n"
        "- Do NOT start with 'Based on the evidence'."
        + _STYLE_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        f"Ranked candidates ({direction}) [structured, normalized]:\n"
        + ("\n".join(evidence_lines) if evidence_lines else "(none captured structurally)")
        + (
            f"\n\nSupplementary raw text passages (check for missing entities):\n"
            + "\n".join(raw_supplement[:20])
            if raw_supplement else ""
        )
        + "\n\nAnswer directly and completely:"
    )
    raw, usage = call_model(system, user, model=SOLVER_MODEL, temperature=0.1, max_tokens=MAX_TOKENS_SOLVER)
    return raw.strip(), usage


# ── Absence helpers ───────────────────────────────────────────────────────────

def _search_phrase_variants(store: KnowledgeStore, phrase: str) -> bool:
    """
    Search for phrase and its hyphen↔space variants.
    "four-day workweek" and "four day workweek" are treated as equivalent.
    Returns True if ANY variant is found in the document.
    """
    if store.phrase_search(phrase, limit=1):
        return True
    if '-' in phrase:
        if store.phrase_search(phrase.replace('-', ' '), limit=1):
            return True
    if ' ' in phrase:
        if store.phrase_search(phrase.replace(' ', '-'), limit=1):
            return True
    return False


# ── Absence option parser (Idea A) ───────────────────────────────────────────

def _parse_absence_options(q_txt: str) -> List[str]:
    """
    Extract explicit multiple-choice options directly from an absence question.

    Detects three surface patterns:

      1. Em/en-dash delimited  "— A, B, C, and D —"
      2. Of-clause             "Of A, B, C, and D, which …"
      3. Colon-delimited       "…: A, B, C, or D?"

    Returns [] when no list is detected so the caller falls back to
    Phase 1's topic_keywords.  A sanity gate (2–6 options, each ≥ 3 chars)
    rejects false positives.
    """
    q = q_txt.strip()
    raw_list = ""

    # Pattern 1 — em-dash or en-dash delimited list
    m = re.search(r'[—–]\s*(.+?)\s*[—–]', q)
    if m:
        raw_list = m.group(1).strip()

    # Pattern 2 — "Of A, B, C, and D, which …"
    if not raw_list:
        m = re.search(r'\bOf\s+(.+?),\s+which\b', q, re.IGNORECASE)
        if m:
            raw_list = m.group(1).strip()

    # Pattern 3 — colon-delimited list at the tail of the sentence "…: A, B, C, or D?"
    if not raw_list:
        m = re.search(r':\s*([A-Za-z][^:]+?)\??$', q)
        if m:
            raw_list = m.group(1).strip()

    if not raw_list:
        return []

    options = _split_conjunction_list(raw_list)
    if 2 <= len(options) <= 6 and all(len(o) >= 3 for o in options):
        return options
    return []


def _split_conjunction_list(text: str) -> List[str]:
    """
    Split "A, B, C, and D" or "A, B, C, or D" into ["A", "B", "C", "D"].

    Handles options that themselves contain 'and'/'or' (e.g. "wartime damage
    and military history") by requiring the conjunction to be preceded by a
    comma: only ", and " or ", or " (with leading comma) marks a boundary.

    Strategy: find the RIGHTMOST ", and " / ", or " → last option separator.
    Everything before it is split on ", "; the tail is the last option.
    """
    text = text.strip().rstrip(',?.')

    pos_and = text.rfind(', and ')
    pos_or  = text.rfind(', or ')

    if pos_and >= pos_or and pos_and >= 0:
        sep_pos, sep_len = pos_and, len(', and ')
    elif pos_or >= 0:
        sep_pos, sep_len = pos_or, len(', or ')
    else:
        return [s.strip() for s in text.split(',') if s.strip()]

    rest      = text[:sep_pos]
    last_item = text[sep_pos + sep_len:].strip()
    items     = [s.strip() for s in rest.split(',') if s.strip()]
    if last_item:
        items.append(last_item)
    return [item for item in items if item]





# ── Absence ───────────────────────────────────────────────────────────────────

def _solve_absence(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    """
    Hybrid approach: objective phrase-search anchors + full document reading.

    Step 1: Parse options from question text (Idea A).
    Step 2: For each option, Python phrase_search its key content words across
            ALL raw chunks — no LLM expansion, no false positives.
            Words absent from every chunk are reported as hard evidence.
    Step 3: Pass both the pre-search results and the full raw document text to
            the LLM. The LLM uses the pre-search anchors to avoid hallucinating
            presence from training knowledge, and uses the full text to judge
            whether words that DO appear are used in the right context.
    """
    # ── Step 1: Parse options ─────────────────────────────────────────────────
    parsed_options = _parse_absence_options(q_txt)
    if parsed_options:
        topic_keywords = parsed_options
        logger.debug("Absence: %d options parsed from q_txt: %s", len(topic_keywords), topic_keywords)
    else:
        _tk_raw = plan.get("topic_keywords", [])
        if isinstance(_tk_raw, list):
            topic_keywords = [str(k).replace("_", " ") for k in _tk_raw if k]
        elif _tk_raw:
            topic_keywords = [str(_tk_raw).replace("_", " ")]
        else:
            topic_keywords = []
        if not topic_keywords:
            topic_keywords = [q_txt]

    # Method 1: synonym variants — demand_planner provides alternative phrasings
    # for each topic so we don't miss topics expressed with different terminology.
    _kv_raw = plan.get("keyword_variants", {})
    keyword_variants: Dict[str, List[str]] = _kv_raw if isinstance(_kv_raw, dict) else {}

    # ── Step 2: Objective phrase-search for each option's key content words ───
    _FW = {
        "the", "a", "an", "of", "for", "with", "in", "at", "by", "and", "or",
        "is", "its", "their", "from", "that", "this", "are", "which", "never",
        "among", "about", "across",
    }
    pre_lines: List[str] = []
    phrase_absent: List[str] = []   # options whose exact phrase is NOT in the document
    for opt in topic_keywords:
        # Fix C: full-phrase search first — most precise signal.
        # e.g. "Natura 2000" may be absent as a phrase even if "natura" appears in Italian.
        full_phrase_hit = _search_phrase_variants(store, opt.lower())

        # Method 1: if main phrase absent, check demand_planner synonym variants
        # before declaring this topic absent. Documents may use different terminology.
        _matched_variant: Optional[str] = None
        if not full_phrase_hit:
            for _variant in keyword_variants.get(opt, [])[:6]:
                _variant_str = str(_variant or "").strip()
                if _variant_str and _search_phrase_variants(store, _variant_str.lower()):
                    full_phrase_hit = True
                    _matched_variant = _variant_str
                    break

        if not full_phrase_hit:
            pre_lines.append(
                f'  "{opt}" → PHRASE-SEARCH: phrase (and hyphen/space variants) absent from document'
            )
            phrase_absent.append(opt)
        else:
            if _matched_variant:
                pre_lines.append(
                    f'  "{opt}" → VARIANT-SEARCH: synonym "{_matched_variant}" found '
                    f'(main phrase absent, but concept present via synonym)'
                )
            else:
                pre_lines.append(
                    f'  "{opt}" → PHRASE-SEARCH: phrase (or variant) found in document'
                )

        # Word-by-word search (supplementary signal)
        content_words = [
            w.strip(".,;:") for w in opt.lower().split()
            if len(w.strip(".,;:")) >= 4 and w.strip(".,;:").lower() not in _FW
        ]
        missing = [w for w in content_words if not store.phrase_search(w, limit=1)]
        found   = [w for w in content_words if store.phrase_search(w, limit=1)]
        if missing and not found:
            pre_lines.append(
                f'  "{opt}" → WORD-SEARCH: ALL key words absent from document'
                f' (searched: {content_words})'
            )
        elif missing:
            pre_lines.append(
                f'  "{opt}" → WORD-SEARCH: some words found {found},'
                f' some NOT found {missing}'
            )
        else:
            pre_lines.append(
                f'  "{opt}" → WORD-SEARCH: all key words found in document {found}'
            )

    # ── Absence A: Hard filter ─────────────────────────────────────────────────
    # Fire only when: (a) exactly one phrase is absent AND
    #                 (b) at least one content word of that phrase is also absent.
    # Condition (b) prevents false positives when the phrase is just differently
    # hyphenated/spaced but the topic IS present (e.g. "four-day workweek" found
    # as "four-day work week" — phrase absent but all content words present).
    llm_candidates = phrase_absent if phrase_absent else topic_keywords

    if len(phrase_absent) == 1:
        opt = phrase_absent[0]
        content_words_check = [
            w.strip(".,;:") for w in opt.lower().split()
            if len(w.strip(".,;:")) >= 4 and w.strip(".,;:").lower() not in _FW
        ]
        absent_content_words = [
            w for w in content_words_check
            if not store.phrase_search(w, limit=1)
        ]
        if absent_content_words or not content_words_check:
            # High-confidence: phrase gone AND content words gone → return directly
            return (
                f"The topic that is never substantively discussed in the document is {opt}.",
                {},
            )
        # Low-confidence: phrase absent but all content words present
        # → topic likely discussed with different hyphenation/spacing → let LLM decide
        llm_candidates = topic_keywords

    confirmed_present = [opt for opt in topic_keywords if opt not in llm_candidates]

    # ── Step 3: Full document text ────────────────────────────────────────────
    MAX_DOC_CHARS = 200_000
    doc_parts: List[str] = []
    total_chars = 0
    for i, chunk in enumerate(store._raw_chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        section = f"[Section {i + 1}]\n{text}"
        if total_chars + len(section) > MAX_DOC_CHARS:
            doc_parts.append("[... remaining sections omitted due to length ...]")
            break
        doc_parts.append(section)
        total_chars += len(section)

    doc_text = "\n\n".join(doc_parts)
    options_str = "\n".join(
        f"  {chr(65 + i)}. {opt}" for i, opt in enumerate(llm_candidates)
    )

    # ── Step 4: LLM decision ─────────────────────────────────────────────────
    pre_filter_note = ""
    if confirmed_present:
        pre_filter_note = (
            "PRE-FILTER (Python phrase-search confirmed these ARE in the document — "
            "already eliminated):\n"
            + "\n".join(f"  ✗ {opt}" for opt in confirmed_present)
            + "\n\n"
        )

    system = (
        "You are reading a non-fiction book and answering a question about its content.\n"
        "The question asks which ONE of several listed topics is NEVER substantively "
        "discussed anywhere in the book.\n\n"
        "OPTIONS shown below have already been pre-filtered: topics whose exact phrase was "
        "found verbatim in the document have been REMOVED. Choose only from the remaining options.\n\n"
        "You are given TWO sources of evidence — use BOTH:\n\n"
        "SOURCE A — SEARCH RESULTS (100% objective, computed by Python):\n"
        "  For each option, two checks are reported:\n"
        "  • PHRASE-SEARCH: literal search for the COMPLETE option phrase in the document.\n"
        "    If the exact phrase is absent, that option's specific named concept is almost\n"
        "    certainly NOT discussed (even if individual words appear in other contexts).\n"
        "    RULE: 'EXACT PHRASE absent' is STRONG evidence the topic is absent.\n"
        "  • WORD-SEARCH: checks each key word individually.\n"
        "    WARNING: individual words can appear in unrelated contexts — a word match alone\n"
        "    does NOT prove the topic is discussed.\n\n"
        "SOURCE B — FULL DOCUMENT TEXT:\n"
        "  Use this to confirm your choice from the remaining candidates.\n\n"
        "INSTRUCTIONS:\n"
        "  1. Choose ONLY from the options listed below (pre-filtered candidates).\n"
        "  2. Among those, pick the ONE that is NEVER substantively discussed.\n"
        "  3. CRITICAL — 'substantively discussed' means the topic is a meaningful subject "
        "     of at least one full paragraph or section. A single passing mention in a "
        "     sentence, a list item, or a footnote does NOT count as substantive discussion.\n"
        "  4. Conversely, if VARIANT-SEARCH found a synonym, that topic IS present — "
        "     do not select it as absent just because the exact phrasing differs.\n"
        "  5. State your answer as a complete sentence naming the absent topic."
        + _STYLE_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        + pre_filter_note
        + f"Remaining candidates (choose from these only):\n{options_str}\n\n"
        f"SOURCE A — Word-Search Results:\n"
        + "\n".join(pre_lines)
        + f"\n\nSOURCE B — Full document text:\n{doc_text}\n\n"
        "Which candidate is NEVER substantively discussed? Answer directly:"
    )

    raw, usage = call_model(
        system, user, model=SOLVER_MODEL, temperature=0.1, max_tokens=MAX_TOKENS_SOLVER
    )
    return raw.strip(), usage


# ── Contradiction hint-phrase generator (Idea A) ─────────────────────────────

def _contradiction_hint_phrases(predicate: Optional[str], q_txt: str) -> List[str]:
    """
    Generate raw-text search phrases to surface contradiction evidence.

    Strategy: predicate + question keywords → specific phrases likely to appear
    verbatim in the book near the contradicting claim.

    d10 (rank)      → "second-highest", "second highest", "highest mountain in", …
    d11 (area_km2)  → "largest national park", "norway's largest", …
    d12 (area_km2)  → "sq miles", "square miles" (unit mismatch)
    """
    pred    = (predicate or "").lower().replace("_", " ")
    q_lower = q_txt.lower()
    hints: List[str] = []

    # Unit-mismatch indicators — ONLY when the question explicitly mentions units/miles.
    # Keeping "sq miles" unconditional caused the d12 answer (Jotunheimen in sq miles)
    # to bleed into d11 evidence, confusing the LLM about which park is larger.
    if any(w in q_lower for w in ("units", "miles", "square", "different unit")):
        hints += ["sq miles", "square miles", "sq mi"]

    # Rank / ordinal contradictions
    if "rank" in pred or any(w in q_lower for w in ("second", "rank", "third", "ordinal")):
        hints += [
            "second-highest", "second highest",
            "highest mountain", "highest peak",
            "third-highest", "third highest",
            # "highest point" surfaces "Highest point: Ben Macdui (m) 1309" from Cairngorms
            # — the counter-evidence that disproves a "second-highest" claim elsewhere.
            "highest point",
        ]

    # Area superlative contradictions
    if "area" in pred or any(w in q_lower for w in ("largest", "biggest", "smallest")):
        hints += [
            "largest in",
            "biggest in",
        ]
        # Country-qualified superlatives — extract proper nouns from the question
        # itself so this works on any document, not just European parks.
        for tok in q_txt.split():
            tok_clean = tok.strip("'s?,.")
            if len(tok_clean) > 3 and tok_clean[0].isupper() and tok_clean.lower() != tok_clean:
                tl = tok_clean.lower()
                hints += [f"{tl}'s largest", f"largest in {tl}"]

    # Visitor count superlatives
    if "visitor" in pred or "visitor" in q_lower:
        hints += ["most visited", "most popular", "highest number of visitors"]

    # Deduplicate while preserving order
    seen_h: set = set()
    unique: List[str] = []
    for h in hints:
        if h not in seen_h:
            seen_h.add(h)
            unique.append(h)
    return unique


# ── Contradiction B: numeric proximity conflict finder ────────────────────────

def _find_proximity_conflicts(store: KnowledgeStore, rel_threshold: float = 0.02) -> List[str]:
    """
    Find pairs of facts where the same subject has two different but very close
    normalized values (within rel_threshold relative difference).

    This catches off-by-one errors across different tables or document sections
    where the predicate names may differ (e.g. employer_count vs employer_obs).
    Returns formatted NEAR-MATCH evidence lines.
    """
    # Group all numeric facts by subject
    by_subject: Dict[str, List] = {}
    for f in (store._facts or []):
        nv = f.get("normalized_value")
        if nv is None:
            continue
        try:
            nv_f = float(nv)
        except (ValueError, TypeError):
            continue
        if nv_f <= 0:
            continue
        subj = (f.get("subject") or "").strip()
        if not subj:
            continue
        by_subject.setdefault(subj, []).append({**f, "_nv": nv_f})

    lines: List[str] = []
    for subj, facts in by_subject.items():
        if len(facts) < 2:
            continue
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                v1, v2 = f1["_nv"], f2["_nv"]
                if v1 == v2:
                    continue
                rel = abs(v1 - v2) / max(abs(v1), abs(v2))
                if rel > rel_threshold:
                    continue
                # Skip exact duplicate records (same chunk + same predicate)
                if (f1.get("chunk_id") == f2.get("chunk_id")
                        and f1.get("predicate") == f2.get("predicate")):
                    continue
                u1 = f"{f1.get('value','')} {f1.get('unit','')}".strip()
                u2 = f"{f2.get('value','')} {f2.get('unit','')}".strip()
                lines.append(
                    f"NEAR-MATCH: {subj}"
                    f"  |  [{f1.get('predicate','')}] {u1}"
                    f"  vs  [{f2.get('predicate','')}] {u2}"
                    f"  (Δ={abs(v1-v2):.4g}, {rel*100:.2f}%)"
                )
    return lines


# ── Contradiction ─────────────────────────────────────────────────────────────

def _solve_contradiction(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    _pred_raw = plan.get("predicate", None)
    predicate = str(_pred_raw or "").strip() or None

    # ── Track A: numerical same-entity conflicts ──────────────────────────────
    contradictions = store.find_contradictions(predicate)
    if not contradictions and predicate:
        contradictions = store.find_contradictions(None)
        if contradictions and predicate:
            pred_words = {w for w in predicate.lower().replace("_", " ").split() if len(w) > 3}
            contradictions = {
                k: v for k, v in contradictions.items()
                if not pred_words or any(w in k for w in pred_words)
            } or contradictions

    conflict_lines = []
    for key, facts in list(contradictions.items())[:10]:
        parts = key.split("||")
        subj  = parts[0] if parts else "?"
        pred  = parts[1] if len(parts) > 1 else "?"
        conflict_lines.append(f"CONFLICT: {subj} / {pred}")
        for f in facts[:4]:
            val_str = f"{f.get('value') or ''} {f.get('unit') or ''}".strip()
            conflict_lines.append(f"  chunk={f.get('chunk_id','?')} → {val_str}")

    # ── Track B: unit inconsistency across entities (catches d12-style) ──────
    unit_lines = store.find_unit_outliers(predicate or "area", allowed_types=None)

    # ── Track C: superlative CLAIM keyword search ─────────────────────────────
    claim_lines = store.keyword_search("largest biggest highest second lowest smallest", limit=25)

    # ── Track D: raw sentence / keyword search ────────────────────────────────
    sent_lines = store.sentence_search(q_txt, limit=25)
    kw_lines   = store.keyword_search(q_txt, limit=15)

    # ── Track I: subject-targeted raw scan ────────────────────────────────────
    # Extract proper nouns from the question (e.g. "Sweden", "Italy") and scan
    # raw chunks for sentences containing those nouns + a number.  This surfaces
    # table rows the distiller may not have captured as structured facts.
    _subject_lines: List[str] = []
    _q_proper = [
        w.strip("'\".,?!();:")
        for w in q_txt.split()
        if len(w.strip("'\".,?!();:")) > 3
        and w.strip("'\".,?!();:")[0].isupper()
        and w.strip("'\".,?!();:").lower() not in {
            "which", "what", "where", "when", "who", "how", "find", "does",
            "this", "that", "these", "those", "both", "also", "have", "been",
        }
    ]
    for _noun in _q_proper[:4]:
        for _hit in store.phrase_search(_noun.lower(), limit=8):
            if any(c.isdigit() for c in _hit):
                _subject_lines.append(f"RAW-SUBJECT ({_noun}): {_hit[:250]}")

    # ── Track E: phrase_search for direct contradiction evidence ──────────────
    phrase_hit_lines: List[str] = []
    for _phrase in _contradiction_hint_phrases(predicate, q_txt):
        for _hit in store.phrase_search(_phrase, limit=5):
            phrase_hit_lines.append(f"RAW-HINT ({_phrase!r}): {_hit[:220]}")

    # ── Track F: ALL numeric facts, all predicates (C1) ─────────────────────
    # Dump every numeric fact the distiller captured.  No predicate whitelist —
    # the LLM reads the question and decides which predicate and entity are
    # relevant.  This makes the solver document-agnostic.
    all_fact_lines: List[str] = []
    for f in (store._facts or []):
        val_str = f"{f.get('value', '')} {f.get('unit', '')}".strip()
        if not val_str:
            continue
        pred = f.get("predicate", "")
        nv = f.get("normalized_value", "")
        line = f"ALL-FACTS [{pred}]: {f.get('subject', '')} = {val_str}"
        if nv not in ("", None):
            line += f"  (norm={nv})"
        all_fact_lines.append(line)

    # ── Track G: ALL claims from knowledge store (C1) ─────────────────────────
    # Every claim the distiller captured, unfiltered. Superlative claims like
    # "Hardangervidda is Norway's largest national park" live here and are
    # essential context for spotting area-based contradictions.
    all_claim_lines: List[str] = []
    for c in (store._claims or []):
        text = c.get("text", "")
        if text:
            entity = c.get("entity", "")
            all_claim_lines.append(
                f"CLAIM [{entity}]: {text}" if entity else f"CLAIM: {text}"
            )

    # ── Track H: Numeric proximity conflicts (Contradiction B) ────────────────
    # Finds pairs where the same subject has slightly different values (within 2%)
    # across different facts — catches off-by-one cross-table errors like 408 vs 409.
    proximity_lines = _find_proximity_conflicts(store, rel_threshold=0.02)

    # ── Track J: Entity-focused claim/fact filter ─────────────────────────────
    # When the question names a specific entity (e.g. "Northlight"), the LLM
    # must focus on THAT entity's claims and facts — not pick an unrelated
    # contradiction from the document.  Put entity-specific lines first so they
    # dominate the LLM's attention window.
    _entity_claims: List[str] = []
    _entity_facts: List[str] = []
    if _q_proper:
        _q_proper_lower = [n.lower() for n in _q_proper[:4]]
        for line in all_claim_lines:
            if any(n in line.lower() for n in _q_proper_lower):
                _entity_claims.append(f"FOCUS-CLAIM: {line}")
        for line in all_fact_lines:
            if any(n in line.lower() for n in _q_proper_lower):
                _entity_facts.append(f"FOCUS-FACT: {line}")

    # Evidence assembly — entity-focused lines first, then phrase hits, then rest.
    evidence: List[str] = []
    seen: set = set()
    for line in (
        _entity_claims
        + _entity_facts
        + phrase_hit_lines
        + proximity_lines
        + _subject_lines
        + unit_lines
        + all_fact_lines
        + all_claim_lines
        + conflict_lines
        + claim_lines
        + kw_lines
        + sent_lines
    ):
        key = line[:80]
        if key not in seen:
            seen.add(key)
            evidence.append(line)

    if not evidence:
        evidence = ["No contradictions detected in document scan."]

    system = (
        "You are answering a contradiction question — finding conflicting or inconsistent "
        "information within the same document.\n\n"
        "EVIDENCE LINE TYPES:\n"
        "  FOCUS-CLAIM        : claim about the SPECIFIC entity named in the question. "
        "                       Read these FIRST — the contradiction almost certainly involves this entity.\n"
        "  FOCUS-FACT         : numeric fact about the SPECIFIC entity named in the question.\n"
        "  RAW-HINT (phrase)  : verbatim sentence from the document. Highest priority.\n"
        "  RAW-SUBJECT (noun) : verbatim sentence containing a proper noun from the question\n"
        "                       and a number — use these to find exact values for the subject.\n"
        "  NEAR-MATCH         : same subject with two slightly different numeric values.\n"
        "  UNIT-GROUP [unit]  : entities reporting a value in that unit.\n"
        "  ALL-FACTS [pred]   : complete list of every entity's value for that predicate.\n"
        "                       Use this to spot superlative claims that are factually wrong.\n"
        "  CLAIM [entity]     : a claim the distiller captured from the document.\n"
        "  CONFLICT           : same entity with two different values across chunks.\n\n"
        "IMPORTANT: The question asks about a SPECIFIC type of contradiction (rank, area, age, etc.).\n"
        "  Read the question carefully to identify WHICH metric or claim is being questioned,\n"
        "  then look for evidence that directly supports or refutes that specific claim.\n"
        "  Do NOT pivot to a different type of contradiction just because more evidence exists for it.\n\n"
        "FOUR TYPES of contradictions to look for:\n\n"
        "1. UNIT INCONSISTENCY:\n"
        "   If UNIT-GROUP shows one entity reporting a value in a different unit from all others, "
        "   that IS the contradiction. State the entity and its unit.\n\n"
        "2. SUPERLATIVE CLAIM vs FACT:\n"
        "   A CLAIM says 'X is the largest/highest/oldest in [group]' but ALL-FACTS "
        "   shows another entity has a larger/higher/older value in the same group. "
        "   Read the claim carefully to identify which group the comparison is restricted to "
        "   (e.g. a country, a region, a category), then check ALL-FACTS for entities in that "
        "   same group that exceed the claimed value.\n\n"
        "3. NEAR-MATCH VALUE INCONSISTENCY:\n"
        "   NEAR-MATCH lines show the same subject with two slightly different numeric values\n"
        "   (within 2% relative). Even a difference of 1 is a real contradiction when the same\n"
        "   entity is reported differently across document sources (e.g. 408 vs 409 companies).\n"
        "   State the subject, both values, and their source predicates/units.\n\n"
        "4. SAME ENTITY, DIFFERENT VALUE:\n"
        "   CONFLICT lines show the same entity reported with two different values across "
        "   different document sections.\n\n"
        "4. CROSS-TABLE VALUE INCONSISTENCY:\n"
        "   ALL-FACTS or CONFLICT lines show the same subject with the same predicate but "
        "   different numeric values, where the unit field reveals they come from different "
        "   tables (e.g., 'observations [Table 5.1]' vs 'observations [Annex Table 5.A.5]').\n"
        "   Even a difference of 1 between two tables is a contradiction. State both table "
        "   names/sources and their exact numeric values for the subject.\n\n"
        "Rules:\n"
        "- Examine ALL evidence types. Use ALL-FACTS to cross-check CLAIM superlatives.\n"
        "- For superlative contradictions: identify the claim, find the contradicting fact, "
        "  name both entities and their values.\n"
        "- For cross-table contradictions: identify the subject, both table sources, "
        "  and both reported values.\n"
        "- Quote verbatim from RAW-HINT lines where possible.\n"
        "- Use ONLY the evidence provided.\n"
        "- VALUE CITATION (critical): Only state specific numeric values if they appear "
        "  verbatim in a NEAR-MATCH, CONFLICT, ALL-FACTS, or RAW-HINT line above. "
        "  Do NOT infer or reconstruct values from context.\n"
        "- SOURCE CITATION (critical): Only name specific table names, section numbers, or "
        "  sources if they appear explicitly in the evidence lines. Do not guess source names.\n"
        "- If you can identify the subject and type of contradiction but the exact numeric "
        "  values are not present in the evidence, state the subject and type clearly and "
        "  acknowledge that the specific figures are not available in the provided evidence."
        + _STYLE_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        f"Evidence:\n"
        + "\n".join(evidence[:150])
        + "\n\nIdentify and describe the contradiction clearly:"
    )
    raw, usage = call_model(system, user, model=SOLVER_MODEL, temperature=0.1, max_tokens=MAX_TOKENS_SOLVER)
    return raw.strip(), usage


# ── Needle ────────────────────────────────────────────────────────────────────

def _solve_needle(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    """
    Two-track search — structured + sentence scan always run in parallel.

    Track A  Structured store → raw paragraph (first non-empty tier)
             Gives typed facts / entity attributes extracted in Phase 2.

    Track B  Sentence-level full document scan (always runs)
             Catches specific facts the Phase 2 distiller missed — the
             "전체 문서를 훑는" guarantee that no verbatim answer is skipped.

    Both tracks' results are merged before LLM synthesis, so the LLM
    always sees both structured knowledge and raw evidence.
    """
    _kw_raw    = plan.get("keywords", [])
    keywords   = [k.replace("_", " ") for k in (_kw_raw if isinstance(_kw_raw, list) else [])]
    plan_query = " ".join(keywords) if keywords else q_txt

    # ── Track A: structured / raw paragraph (first non-empty tier) ───────────
    structured = store.keyword_search(plan_query, limit=20)
    if not structured and keywords:
        structured = store.keyword_search(q_txt, limit=20)
    if not structured:
        structured = store.keyword_search_raw(plan_query, limit=15)
    if not structured and keywords:
        structured = store.keyword_search_raw(q_txt, limit=15)

    # ── Track B: sentence-level full scan (always runs) ──────────────────────
    sentences = store.sentence_search(q_txt, limit=25)

    # ── Merge & deduplicate (sentences first — LLM attends to early context) ──
    seen: set = set()
    evidence: List[str] = []
    for line in (sentences or []) + (structured or []):
        key = line[:80]
        if key not in seen:
            seen.add(key)
            evidence.append(line)

    if not evidence:
        return "This information is not found in the document.", {}

    return _llm_synthesis(q_txt, "needle", evidence, raw=True)


# ── Cross-section ─────────────────────────────────────────────────────────────

def _solve_cross_section(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    _et_raw      = plan.get("entity_types", [])
    entity_types = _et_raw if isinstance(_et_raw, list) else ([str(_et_raw)] if _et_raw else [])
    _kw_raw      = plan.get("keywords",     [])
    _kw_list     = _kw_raw if isinstance(_kw_raw, list) else ([str(_kw_raw)] if _kw_raw else [])
    # Normalize underscores to spaces so "first_ascent" matches "first ascent" in text
    keywords     = [k.replace("_", " ") for k in _kw_list]

    # X3: detect question theme for targeted claim/event filtering
    q_lower = q_txt.lower()
    _is_border_q = any(w in q_lower for w in (
        "border", "international", "cross-border", "shared", "countries",
        "paired", "extend", "transboundary", "twinned", "formally"
    ))
    _is_first_q = any(w in q_lower for w in (
        "first", "earliest", "record", "ascent", "climb", "recorded", "climbing"
    ))

    # Track A: entities of each target type
    entity_lines = []
    for et in entity_types[:3]:
        for e in store.entities_by_type(et)[:50]:
            attrs = e.get("attributes") or {}
            attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:5])
            entry = f"ENTITY [{et}]: {e.get('name','')}"
            if attr_str:
                entry += f"  ({attr_str})"
            entity_lines.append(entry)

    # Track B: keyword search on structured store (finds claims, themes, events, facts)
    kw_query = " ".join(keywords) if keywords else q_txt
    kw_lines = store.keyword_search(kw_query, limit=40)

    # Track C: sentence-level raw scan — always on, catches prose descriptions
    sent_lines = store.sentence_search(kw_query, limit=25)

    # Track D: raw paragraph fallback
    raw_lines: List[str] = []
    if not kw_lines and not sent_lines:
        raw_lines = store.keyword_search_raw(q_txt, limit=15)

    # Track E: phrase_search for specific keywords — catches structured events
    phrase_lines: List[str] = []
    for kw in keywords[:4]:
        for hit in store.phrase_search_variants(kw, limit=4):
            phrase_lines.append(f"RAW: {hit[:200]}")

    # Track F (X3): claims filtered by question theme — document-agnostic keywords only
    all_claim_lines: List[str] = []
    _border_words = ("border", "land bridge", "shared", "twinned", "transboundary",
                     "cross", "international", "paired", "formally")
    _first_words  = ("first", "ascent", "climb", "record", "earliest")
    for c in (store._claims or []):
        text = c.get("text", "")
        if not text:
            continue
        tl = text.lower()
        include = False
        if _is_border_q:
            include = any(w in tl for w in _border_words)
        elif _is_first_q:
            include = any(w in tl for w in _first_words)
        # else: include stays False — don't flood LLM with unrelated claims
        if include:
            entity = c.get("entity", "")
            all_claim_lines.append(
                f"CLAIM [{entity}]: {text}" if entity else f"CLAIM: {text}"
            )

    # Track G (X3): ALL events filtered for first/record questions
    all_event_lines: List[str] = []
    if _is_first_q:
        for ev in (store._events or []):
            desc = ev.get("description", "")
            if not desc:
                continue
            if any(w in desc.lower() for w in _first_words):
                yr   = ev.get("year_text", "")
                ent  = ev.get("entity", "")
                line = f"EVENT ({yr}) [{ent}]: {desc}" if yr else f"EVENT [{ent}]: {desc}"
                all_event_lines.append(line)

    # Track H: events for entities explicitly named in the question.
    # Cross-section questions often name specific entities (e.g. a station, a park, a person).
    # Fetch ALL events for those entities regardless of keyword filter — this catches
    # establishment years, founding dates, and other temporal facts the keyword filters miss.
    q_proper: List[str] = [
        w.strip("'\".,?!();:").lower()
        for w in q_txt.split()
        if len(w.strip("'\".,?!();:")) > 3
        and w.strip("'\".,?!();:")[0].isupper()
        and w.strip("'\".,?!();:").lower() not in {
            "which", "what", "where", "when", "who", "how", "this", "that",
            "these", "those", "both", "also", "does", "have", "been",
        }
    ]
    q_entity_events: List[str] = []
    if q_proper:
        for ev in (store._events or []):
            ent_low  = (ev.get("entity")      or "").lower()
            desc_low = (ev.get("description") or "").lower()
            combined = ent_low + " " + desc_low
            if any(pn in combined for pn in q_proper):
                yr   = ev.get("year_text", "")
                ent  = ev.get("entity", "")
                desc = ev.get("description", "")
                line = f"EVENT ({yr}) [{ent}]: {desc}" if yr else f"EVENT [{ent}]: {desc}"
                q_entity_events.append(line)

    # Merge deduplicated — claim/event tracks first so LLM sees them in early context
    seen: set = set()
    lines: List[str] = []
    for line in (all_claim_lines + all_event_lines + q_entity_events +
                 entity_lines + kw_lines + sent_lines + raw_lines + phrase_lines):
        key = line[:80]
        if key not in seen:
            seen.add(key)
            lines.append(line)

    system = (
        "You are answering a cross-section question — one that requires connecting facts "
        "spread across multiple parts of the document.\n"
        "Evidence line types:\n"
        "  CLAIM [entity]: a structured claim about that entity extracted from the document.\n"
        "  EVENT (year) [entity]: a dated event extracted from the document.\n"
        "  ENTITY [type]: a named entity with attributes.\n"
        "  RAW: verbatim sentence from the source text.\n"
        "Rules:\n"
        "- MULTI-CONDITION RULE: Parse the question carefully and identify EVERY distinct "
        "  condition it imposes. Only accept an item as the answer if it satisfies ALL conditions "
        "  simultaneously — not just one or two. Do NOT answer about a single condition in "
        "  isolation.\n"
        "- EVIDENCE SYNTHESIS: Cross-section questions by design require connecting evidence "
        "  from DIFFERENT parts of the document. Evidence for condition A may come from a "
        "  different evidence line than evidence for condition B — this is expected. Synthesize "
        "  across all evidence lines to find the entity that satisfies every condition.\n"
        "- ANTI-HALLUCINATION: Only name specific entities, numeric values, dates, or table "
        "  sources that appear in the evidence lines above. Do NOT invent connections, policies, "
        "  or facts that are not in the evidence.\n"
        "- For border/international questions: identify every entity or location that crosses or "
        "  is paired across a national border, and name all countries involved.\n"
        "- For first/record questions: find the specific date, person, and location named.\n"
        "- List every qualifying entity; do not omit any that appear in the evidence.\n"
        "- If truly no entity meets all conditions, briefly explain which conditions could not "
        "  be met, then give the best partial answer from the available evidence."
        + _STYLE_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        f"Evidence ({len(lines)} lines):\n"
        + "\n".join(lines[:120] if lines else ["No evidence found in document scan."])
        + "\n\nAnswer completely:"
    )
    raw, usage = call_model(system, user, model=SOLVER_MODEL, temperature=0.1, max_tokens=MAX_TOKENS_SOLVER)
    return raw.strip(), usage


# ── Global synthesis ──────────────────────────────────────────────────────────

def _solve_global_synthesis(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    summary = store.to_summary_text(max_tokens=6000)

    system = (
        "You are answering a global synthesis question about a long document.\n"
        "The question asks for an overall characterisation, pattern, or broad summary.\n"
        "Rules:\n"
        "- Use ONLY the knowledge summary provided below.\n"
        "- Identify patterns, themes, and broad characteristics.\n"
        "- Give a comprehensive answer in 3-7 sentences.\n"
        "- Do not use outside knowledge."
        + _STYLE_RULE
    )
    user = (
        f"Question: {q_txt}\n\n"
        f"Document knowledge summary:\n{summary}\n\nAnswer comprehensively:"
    )
    raw, usage = call_model(
        system, user,
        model=SOLVER_MODEL,
        temperature=TEMPERATURE_ANSWER,
        max_tokens=MAX_TOKENS_SOLVER,
    )
    return raw.strip(), usage


# ── Generic fallback ──────────────────────────────────────────────────────────

def _solve_generic(q_txt: str, plan: Dict, store: KnowledgeStore) -> Tuple[str, Dict]:
    kw_lines = store.keyword_search(q_txt, limit=30)
    return _llm_synthesis(q_txt, "general", kw_lines or ["No evidence found."])


# ── LLM synthesis helper ──────────────────────────────────────────────────────

def _llm_synthesis(
    q_txt:    str,
    q_cat:    str,
    evidence: List[str],
    raw:      bool = False,
) -> Tuple[str, Dict]:
    evidence_block = "\n".join(evidence[:50]) if evidence else "No evidence found."
    label = "Raw text passages from original document" if raw else "Evidence from structured document scan"

    system = (
        "You are an expert analyst synthesizing document evidence to answer a question.\n"
        "Use ONLY the evidence provided. Do not use outside knowledge.\n"
        "If the answer cannot be determined from the evidence, respond with exactly: "
        "'This information is not found in the document.'\n"
        "Do NOT guess or invent entities or values not in the evidence.\n"
        "Write a clear, concise, complete answer in 1-5 sentences.\n"
        "Do NOT start with 'Based on the evidence'."
        + _STYLE_RULE
        + _GROUNDING_RULE
    )
    user = (
        f"Question ({q_cat}): {q_txt}\n\n"
        f"{label}:\n{evidence_block}\n\nAnswer:"
    )
    result, usage = call_model(
        system, user,
        model=SOLVER_MODEL,
        temperature=0.1,
        max_tokens=MAX_TOKENS_SOLVER,
    )
    return result.strip(), usage


# ── Unit normalization ────────────────────────────────────────────────────────

_AREA_TO_KM2 = {
    "ha": 0.01, "hectare": 0.01, "hectares": 0.01,
    "acre": 0.004047, "acres": 0.004047,
    "mile": 2.58999, "miles": 2.58999, "mi": 2.58999,
    "mi2": 2.58999, "sq mi": 2.58999, "sq mile": 2.58999,
    "km2": 1.0, "km²": 1.0, "sq km": 1.0,
}


def _normalize_area(fact: Dict) -> Dict:
    unit = (fact.get("unit") or "").lower().strip()
    nv   = fact.get("normalized_value")
    if nv is None:
        return fact
    for suffix, factor in _AREA_TO_KM2.items():
        if suffix in unit:
            fact = dict(fact)
            fact["normalized_value"] = float(nv) * factor
            return fact
    return fact


# ── Debug / Inspection ────────────────────────────────────────────────────────

def debug_solve_question(
    question:    Dict[str, Any],
    store:       KnowledgeStore,
    demand_plan: Dict[str, Any],
) -> Tuple[str, Dict]:
    """
    Diagnostic wrapper around solve_question.

    Prints for each question:
      1. Demand Plan  — what Phase 1 planned for this question
      2. Store Preview — what the KnowledgeStore actually holds
      3. Generated Answer + token usage

    Use from a notebook cell:
        from src_v4.solvers import debug_solve_question
        debug_solve_question(question, store, demand_plan)
    """
    q_id  = question.get("id", "")
    q_cat = question.get("category", "needle")
    q_txt = question.get("question", "").strip()
    plan  = demand_plan.get("question_plans", {}).get(q_id, {})

    if not isinstance(plan, dict):
        plan = plan[0] if isinstance(plan, list) and plan else {}

    SEP = "=" * 65

    # ── 1. Demand Plan ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"[{q_id}]  category={q_cat}")
    print(f"Q: {q_txt}")
    print(f"\n▶ Demand Plan  (question_plans['{q_id}']):")
    if plan:
        for k, v in plan.items():
            print(f"    {k:22s}: {v}")
    else:
        print("    (없음 — generic fallback 사용)")

    # ── 2. KnowledgeStore Evidence Preview ───────────────────────────────────
    print(f"\n▶ KnowledgeStore Evidence Preview:")
    _print_store_preview(q_cat, plan, store, q_txt)

    # ── 3. Solve → Answer ─────────────────────────────────────────────────────
    print(f"\n▶ Generated Answer:")
    answer_text, usage = solve_question(question, store, demand_plan)
    print(answer_text)
    p_tok = usage.get("prompt_tokens",     0) or 0
    c_tok = usage.get("completion_tokens", 0) or 0
    print(f"\n  tokens: in={p_tok:,}  out={c_tok:,}")
    print(SEP)

    return answer_text, usage


def _print_store_preview(
    q_cat: str, plan: Dict, store: KnowledgeStore, q_txt: str
) -> None:
    """Print relevant KnowledgeStore contents for the given question type."""

    if q_cat == "aggregation":
        entity_type  = plan.get("entity_type", "")
        filter_key   = plan.get("filter_key",  "")
        filter_value = plan.get("filter_value","")
        predicate    = plan.get("predicate",   "")
        pred         = predicate or filter_key
        if entity_type:
            entities = store.entities_by_type(entity_type)
            print(f"    [Track A] entities_by_type({entity_type!r}): {len(entities)} entities")
            for e in entities[:6]:
                attrs = list((e.get("attributes") or {}).keys())[:4]
                print(f"      - {e.get('name','')}  attrs={attrs}")
            if len(entities) > 6:
                print(f"      ... and {len(entities) - 6} more")
        if pred:
            facts = store.facts_by_predicate(pred)
            print(f"    [Track B] facts_by_predicate({pred!r}): {len(facts)} facts")
            for f in facts[:4]:
                print(f"      - {f.get('subject','')[:30]} / {f.get('predicate')} / {f.get('value')} (norm={f.get('normalized_value')})")
        sent = store.sentence_search(plan.get("_q_txt", filter_value or ""), limit=3)
        print(f"    [Track D] sentence_search: {len(sent)} sentences")

    elif q_cat == "superlative":
        predicate = plan.get("predicate", "")
        direction = plan.get("direction", "max")
        if predicate:
            facts = [f for f in store.facts_by_predicate(predicate)
                     if f.get("normalized_value") is not None]
            ranked = sorted(facts, key=lambda f: float(f["normalized_value"]),
                            reverse=(direction == "max"))
            print(f"    facts_by_predicate({predicate!r}): {len(facts)} with normalized value  (direction={direction})")
            for f in ranked[:5]:
                print(f"      - {f.get('subject',''):<35s} {f.get('value','')} {f.get('unit','')}  "
                      f"(norm={f.get('normalized_value')})")

    elif q_cat == "needle":
        keywords = plan.get("keywords", [])
        query    = " ".join(keywords) if keywords else q_txt
        hits     = store.keyword_search(query, limit=5)
        print(f"    [Track A] keyword_search  keywords={keywords}")
        print(f"      → {len(hits)} structured hits")
        for h in hits[:3]:
            print(f"        {h[:95]}")
        sentences = store.sentence_search(q_txt, limit=5)
        print(f"    [Track B] sentence_search")
        print(f"      → {len(sentences)} sentences found")
        for s in sentences[:3]:
            print(f"        {s[:95]}")

    elif q_cat == "absence":
        topics = plan.get("topic_keywords", [])
        print(f"    topic_keywords: {topics}")
        for t in (topics or [q_txt])[:4]:
            ev = store.check_topic(t)
            kw = store.keyword_search(t, limit=3) or []
            status = "PRESENT" if (ev or kw) else "ABSENT"
            print(f"      [{status}]  check_topic({t!r}): {len(ev)} items  "
                  f"| keyword_search: {len(kw)} hits")

    elif q_cat == "contradiction":
        predicate = plan.get("predicate", None)
        contradictions = store.find_contradictions(predicate) or store.find_contradictions(None)
        print(f"    find_contradictions({predicate!r}): {len(contradictions)} conflict group(s)")
        for key, facts in list(contradictions.items())[:4]:
            parts = key.split("||")
            subj  = parts[0] if parts else "?"
            pred  = parts[1] if len(parts) > 1 else "?"
            print(f"      CONFLICT: {subj} / {pred}")
            for f in facts[:2]:
                print(f"        source={f.get('chunk_id','?')} → "
                      f"{f.get('value','')} {f.get('unit','')}")

    elif q_cat == "cross_section":
        entity_types = plan.get("entity_types", [])
        keywords     = plan.get("keywords",     [])
        for et in entity_types[:3]:
            entities = store.entities_by_type(et)
            print(f"    entities_by_type({et!r}): {len(entities)} entities")
            for e in entities[:3]:
                print(f"      - {e.get('name','')}")
        if keywords:
            hits = store.keyword_search(" ".join(keywords), limit=5)
            print(f"    keyword_search({keywords[:3]}): {len(hits)} hits")
            for h in hits[:3]:
                print(f"      {h[:95]}")

    elif q_cat == "global_synthesis":
        stats = store.stats()
        print(f"    Store: {stats['entities']} entities | {stats['facts']} facts | "
              f"{stats['events']} events | {stats['themes']} themes")

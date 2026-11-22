"""
Main pipeline orchestrator — Distill-and-Solve (pipeline_v4).

Phase 1  Demand Planner         1 LLM call, cached
         Build demand_plan: entity_types, predicates, topics, per-question solving hints

Phase 2  Chunk Distillation     1 LLM call per chunk, parallel, cached
         All chunks processed → KnowledgeStore
         Raw chunk text stored for needle fallback

Phase 3  Solvers                Python-first, LLM synthesis where needed
         aggregation  : entities_by_type → filter → count → LLM format
         superlative  : facts_by_predicate → max/min normalized_value → LLM format
         absence      : check_topic → assess presence/absence → LLM format
         contradiction: find_contradictions → LLM explain
         needle       : keyword_search → LLM; FALLBACK: keyword_search_raw → LLM
         cross_section: collect related evidence → LLM synthesis
         global_synthesis: to_summary_text → LLM synthesis
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .chunk_distiller import distill_chunk
from .chunker import build_chunks, validate_coverage
from .demand_planner import build_demand_plan
from .document_inventory import scan_document_inventory, inventory_summary
from .document_parser import parse_document
from .knowledge_store import KnowledgeStore
from .question_classifier import classify_questions_batch
from .question_loader import load_questions
from .result_writer import save_results
from .solvers import solve_question, debug_solve_question
from .universal_chunker import build_section_chunks, validate_section_coverage
from .config import (
    DISTILLATION_CONCURRENCY,
    CHUNK_MANIFEST_FILE, STORE_FILE,
    OUTPUT_DIR, TOKEN_USAGE_FILE,
    CORE_CHUNK_WORDS, CONTEXT_WORDS_BEFORE, CONTEXT_WORDS_AFTER,
    USE_SECTION_CHUNKER, SECTION_MIN_WORDS, SECTION_MAX_WORDS, SECTION_CONTEXT_WORDS,
    TOKEN_BUDGET_INPUT,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    document_path:      str,
    yaml_path:          str,
    question_ids:       Optional[List[str]] = None,
    verbose:            bool                = True,
    debug:              bool                = False,
    force_replan:       bool                = False,
    use_chunk_cache:    bool                = True,
    concurrency:        Optional[int]       = None,
    use_section_chunks: Optional[bool]      = None,
    load_store:         Optional[str]       = None,
) -> List[Dict[str, Any]]:
    """
    Run the full Distill-and-Solve pipeline.

    load_store: path to a previously saved KnowledgeStore JSON (skips Phase 2 distillation).
    Returns list of result dicts; also writes answers.json and answers_debug.json.
    """

    def log(msg: str):
        if verbose:
            print(msg)

    conc    = concurrency        if concurrency        is not None else DISTILLATION_CONCURRENCY
    use_sec = use_section_chunks if use_section_chunks is not None else USE_SECTION_CHUNKER
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("\n" + "=" * 70)
    log("  Distill-and-Solve (pipeline_v4)")
    log("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # Preparation
    # ═══════════════════════════════════════════════════════════════════
    text    = parse_document(document_path)
    n_words = len(text.split())

    # Proposal C — Document Inventory Scanner (pure regex, no LLM, runs once)
    inventory = scan_document_inventory(text)
    log(f"  Inventory  : {inventory_summary(inventory)}")

    if use_sec:
        chunks = build_section_chunks(
            text,
            min_section_words=SECTION_MIN_WORDS,
            max_section_words=SECTION_MAX_WORDS,
            context_words=SECTION_CONTEXT_WORDS,
        )
        body_words = sum(c["core_word_count"] for c in chunks)
        validate_section_coverage(chunks, body_words)
        chunk_mode = f"section-aware  min={SECTION_MIN_WORDS:,}  max={SECTION_MAX_WORDS:,} words"
    else:
        chunks = build_chunks(
            text,
            core_chunk_words=CORE_CHUNK_WORDS,
            context_before=CONTEXT_WORDS_BEFORE,
            context_after=CONTEXT_WORDS_AFTER,
        )
        validate_coverage(chunks, n_words)
        chunk_mode = f"fixed-count  ~{CORE_CHUNK_WORDS:,} words each"

    n_chunks = len(chunks)
    _save_chunk_manifest(chunks)

    questions = load_questions(yaml_path)
    if question_ids:
        questions = [q for q in questions if q.get("id") in question_ids]

    # Auto-classify questions that have no category field
    unclassified = [q for q in questions if not q.get("category")]
    if unclassified:
        log(f"  Classifying {len(unclassified)} question(s) without a category…")
        preds = classify_questions_batch(unclassified)
        for q in unclassified:
            q["category"] = preds.get(q["id"], "needle")
        log(f"  Classification done: " +
            ", ".join(f"{t}={sum(1 for q in questions if q.get('category') == t)}"
                      for t in ["aggregation", "superlative", "absence",
                                 "contradiction", "cross_section", "global_synthesis", "needle"]
                      if any(q.get("category") == t for q in questions)))

    n_q = len(questions)

    log(f"Document   : {document_path}")
    log(f"Words      : {n_words:,}")
    log(f"Chunks     : {n_chunks}  ({chunk_mode})")
    log(f"Questions  : {n_q}")
    log(f"Concurrency: {conc}  (1 LLM call/chunk = {n_chunks} distill calls)")
    log("")

    if n_q == 0:
        log("No questions to process.")
        return []

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1 — Demand Planner
    # ═══════════════════════════════════════════════════════════════════
    log("PHASE 1 — Building demand plan…")
    t_p1 = time.time()

    demand_plan, plan_usage = build_demand_plan(questions, force=force_replan)

    p1_in  = max(plan_usage.get("prompt_tokens",     0), 0)
    p1_out = max(plan_usage.get("completion_tokens", 0), 0)

    log(f"  entity_types     : {len(demand_plan.get('entity_types', []))}")
    log(f"  numeric_predicates: {len(demand_plan.get('numeric_predicates', []))}")
    log(f"  theme_topics     : {len(demand_plan.get('theme_topics', []))}")
    log(f"  Tokens           : in={p1_in:,}  out={p1_out:,}")
    log(f"  Phase 1 done in {time.time() - t_p1:.0f}s")

    # ── Idea 3: strip absence topic_keywords from theme_topics ────────────────
    # The distiller is primed by theme_topics to extract those themes — if absence
    # question keywords are listed there, the distiller hallucinates theme entries
    # even when the text never mentions them, making absence detection unreliable.
    # We filter them out of the in-memory demand_plan used for Phase 2 distillation.
    # (demand_plan.json on disk is unchanged; only the distiller prompt changes,
    # which invalidates the chunk_distills cache → triggers clean re-distillation.)
    _absence_kws: set = set()
    for _qp in demand_plan.get("question_plans", {}).values():
        if isinstance(_qp, dict) and _qp.get("solver") == "absence":
            for _kw in (_qp.get("topic_keywords") or []):
                for _w in str(_kw or "").lower().replace("_", " ").split():
                    if len(_w) > 3:
                        _absence_kws.add(_w)
    if _absence_kws:
        _orig_themes = demand_plan.get("theme_topics", [])
        _filtered    = [
            t for t in _orig_themes
            if not any(_w in str(t).lower().replace("_", " ") for _w in _absence_kws)
        ]
        demand_plan  = {**demand_plan, "theme_topics": _filtered}
        log(f"  [Idea 3] Removed {len(_orig_themes) - len(_filtered)} absence-related "
            f"theme(s) from distillation prompt  ({len(_filtered)} remain)\n")
    else:
        log("")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 — Chunk Distillation → KnowledgeStore
    # ═══════════════════════════════════════════════════════════════════
    if load_store and os.path.exists(load_store):
        log(f"PHASE 2 — Loading KnowledgeStore from {load_store} (skipping distillation)…")
        store = KnowledgeStore.load(load_store)
        store.set_inventory(inventory)
        p2_in = p2_out = p2_max_in = 0
        log(f"  Loaded: {store.stats()}\n")
    else:
        log(f"PHASE 2 — Distilling {n_chunks} chunk(s)  (concurrency={conc})…")
        t_p2 = time.time()

        store     = KnowledgeStore()
        p2_in     = 0
        p2_out    = 0
        p2_max_in = 0

        if conc <= 1:
            for chunk in chunks:
                distilled, usage = distill_chunk(chunk, demand_plan, use_cache=use_chunk_cache)
                store.ingest(distilled, chunk["chunk_id"], raw_text=chunk.get("core_content", ""))

                call_in  = max(usage.get("prompt_tokens",     0), 0)
                call_out = max(usage.get("completion_tokens", 0), 0)
                p2_in   += call_in
                p2_out  += call_out
                p2_max_in = max(p2_max_in, call_in)

                label = chunk.get("section_heading", "")[:35]
                warn  = f"  ⚠ {call_in:,}/{TOKEN_BUDGET_INPUT:,}" if call_in >= TOKEN_BUDGET_INPUT else ""
                log(
                    f"  {chunk['chunk_id']}  [{label}]  "
                    f"e={len(distilled.get('entities',[]))} "
                    f"f={len(distilled.get('facts',[]))} "
                    f"tokens_in={call_in:,}{warn}"
                )
        else:
            results_map: Dict[int, Any] = {}

            with ThreadPoolExecutor(max_workers=conc) as pool:
                future_map = {
                    pool.submit(distill_chunk, c, demand_plan, use_chunk_cache): c
                    for c in chunks
                }
                for future in as_completed(future_map):
                    chunk = future_map[future]
                    try:
                        distilled, usage = future.result()
                    except Exception as exc:
                        logger.error("Distillation failed %s: %s", chunk["chunk_id"], exc)
                        distilled = {"entities": [], "facts": [], "events": [], "claims": [], "themes": []}
                        usage     = {}
                    results_map[chunk["chunk_index"]] = (chunk, distilled, usage)
                    log(f"  {chunk['chunk_id']}  done  "
                        f"e={len(distilled.get('entities',[]))} f={len(distilled.get('facts',[]))}")

            # Ingest in document order
            for i in range(n_chunks):
                chunk, distilled, usage = results_map[i]
                store.ingest(distilled, chunk["chunk_id"], raw_text=chunk.get("core_content", ""))
                call_in  = max(usage.get("prompt_tokens",     0), 0)
                call_out = max(usage.get("completion_tokens", 0), 0)
                p2_in   += call_in
                p2_out  += call_out
                p2_max_in = max(p2_max_in, call_in)

        store.set_inventory(inventory)
        store.save(STORE_FILE)
        stats = store.stats()
        log(
            f"  Store: {stats['entities']} entities | {stats['facts']} facts | "
            f"{stats['events']} events | {stats['themes']} themes"
        )
        log(
            f"  Phase 2 done in {time.time() - t_p2:.0f}s  |  "
            f"chunks={n_chunks}  tokens_in={p2_in:,}  tokens_out={p2_out:,}\n"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3 — Solve questions
    # ═══════════════════════════════════════════════════════════════════
    log(f"PHASE 3 — Answering {n_q} question(s)…")
    t_p3 = time.time()

    all_results: List[Dict[str, Any]] = []
    p3_in = p3_out = 0

    for q_idx, question in enumerate(questions):
        q_id  = question["id"]
        q_cat = question.get("category", "needle")
        log(f"  [{q_idx + 1}/{n_q}] {q_id}  ({q_cat})")

        t_q = time.time()
        try:
            if debug:
                answer_text, usage = debug_solve_question(question, store, demand_plan)
            else:
                answer_text, usage = solve_question(question, store, demand_plan)
            status = "answered"
        except Exception as exc:
            logger.error("solve_question failed for %s: %s", q_id, exc)
            answer_text = f"[Error: {str(exc)[:200]}]"
            usage  = {}
            status = "error"

        elapsed = round(time.time() - t_q, 2)
        p_tok   = usage.get("prompt_tokens",     0) or 0
        c_tok   = usage.get("completion_tokens", 0) or 0
        p3_in  += p_tok
        p3_out += c_tok

        all_results.append({
            "question_id":       q_id,
            "question_type":     q_cat,
            "question":          question.get("question", ""),
            "answer":            (answer_text or "").strip(),
            "status":            status,
            "safe_to_answer":    True,
            "token_usage":       {"input_tokens": p_tok, "output_tokens": c_tok},
            "processing_time_s": elapsed,
        })
        log(f"    {(answer_text or '')[:110]}…")

    log(f"  Phase 3 done in {time.time() - t_p3:.0f}s\n")

    # ═══════════════════════════════════════════════════════════════════
    # Summary + save
    # ═══════════════════════════════════════════════════════════════════
    total_in    = p1_in  + p2_in  + p3_in
    total_out   = p1_out + p2_out + p3_out
    over_budget = p2_max_in >= TOKEN_BUDGET_INPUT

    log("=" * 70)
    log("TOKEN USAGE SUMMARY")
    log(f"  Phase 1 (plan  ×1)   : in={p1_in:>8,}  out={p1_out:>7,}")
    log(f"  Phase 2 (distill×1)  : in={p2_in:>8,}  out={p2_out:>7,}  chunks={n_chunks}")
    log(f"  Phase 2 max/chunk    : {p2_max_in:,} / {TOKEN_BUDGET_INPUT:,}"
        + ("  ⚠ OVER BUDGET" if over_budget else "  ✓ OK"))
    log(f"  Phase 3 (solve)      : in={p3_in:>8,}  out={p3_out:>7,}")
    log(f"  TOTAL                : in={total_in:>8,}  out={total_out:>7,}")
    log("")
    log("Saving results…")

    paths = save_results(
        all_results,
        document_path=document_path,
        n_chunks=n_chunks,
        total_usage={"input_tokens": total_in, "output_tokens": total_out},
    )

    _save_token_usage(
        p1_in=p1_in,   p1_out=p1_out,
        p2_in=p2_in,   p2_out=p2_out,   p2_calls=n_chunks, p2_max_in=p2_max_in,
        p3_in=p3_in,   p3_out=p3_out,   p3_calls=n_q,
        total_in=total_in, total_out=total_out,
    )

    log(f"  Submission : {paths['submission']}")
    log(f"  Debug      : {paths['debug']}")
    log(f"  Store      : {STORE_FILE}")
    log("=" * 70 + "\n")

    return all_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_chunk_manifest(chunks: List[Dict[str, Any]]) -> None:
    manifest = [
        {
            "chunk_id":        c["chunk_id"],
            "chunk_index":     c["chunk_index"],
            "core_word_start": c["core_word_start"],
            "core_word_end":   c["core_word_end"],
            "core_word_count": c["core_word_count"],
            "section_heading": c.get("section_heading", ""),
        }
        for c in chunks
    ]
    with open(CHUNK_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _save_token_usage(
    p1_in: int, p1_out: int,
    p2_in: int, p2_out: int, p2_calls: int, p2_max_in: int,
    p3_in: int, p3_out: int, p3_calls: int,
    total_in: int, total_out: int,
) -> None:
    record = {
        "token_budget_per_call": TOKEN_BUDGET_INPUT,
        "phase1": {"input": p1_in,  "output": p1_out, "calls": 1,        "llm_calls_per_chunk": "n/a"},
        "phase2": {
            "input": p2_in, "output": p2_out,
            "calls": p2_calls, "llm_calls_per_chunk": 1,
            "max_input_per_chunk": p2_max_in,
            "over_budget": p2_max_in >= TOKEN_BUDGET_INPUT,
        },
        "phase3": {"input": p3_in, "output": p3_out, "calls": p3_calls},
        "total":  {"input": total_in, "output": total_out},
    }
    try:
        with open(TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Could not save token usage: %s", exc)

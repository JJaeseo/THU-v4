"""
app.py — Streamlit demo front-end for pipeline_v4 (Distill-and-Solve).

Run:
    cd pipeline_v4
    streamlit run app.py --server.headless true
"""
import json
import os
import re
import sys
import tempfile

# Force UTF-8 stdout/stderr on Windows (prevents cp949 encode errors with Unicode chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

# ── bootstrap ─────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))
os.chdir(PIPELINE_DIR)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Distill-and-Solve · pipeline_v4",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
TYPE_COLORS: Dict[str, str] = {
    "aggregation":      "#3b82f6",
    "superlative":      "#8b5cf6",
    "absence":          "#ef4444",
    "contradiction":    "#f97316",
    "needle":           "#10b981",
    "cross_section":    "#06b6d4",
    "global_synthesis": "#64748b",
}

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 2px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
    margin-right: 6px;
    vertical-align: middle;
}
.answer-box {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0 4px 0;
    font-size: 14px;
    line-height: 1.75;
    white-space: pre-wrap;
}
.answer-box-ask {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0 4px 0;
    font-size: 14px;
    line-height: 1.75;
    white-space: pre-wrap;
}
.sh {
    font-size: 15px;
    font-weight: 700;
    color: #334155;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 5px;
    margin: 20px 0 10px 0;
}
.welcome-phase {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 22px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────── helpers ──────────────────────────────────────

def _badge(text: str, color: str) -> str:
    return (f'<span class="badge" style="background:{color};color:#fff;">'
            f'{text}</span>')


def type_badge(qtype: str) -> str:
    return _badge(qtype, TYPE_COLORS.get(qtype, "#94a3b8"))


def sh(text: str) -> None:
    st.markdown(f'<p class="sh">{text}</p>', unsafe_allow_html=True)


def load_json_upload(up) -> Optional[Dict]:
    if up is None:
        return None
    up.seek(0)
    try:
        return json.loads(up.read().decode("utf-8"))
    except Exception as e:
        st.error(f"JSON parse error ({up.name}): {e}")
        return None


def load_answer_key(up) -> Optional[List]:
    if up is None:
        return None
    up.seek(0)
    raw = up.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            data = yaml.safe_load(raw.decode("utf-8"))
        except Exception as e:
            st.error(f"Could not parse answer key: {e}")
            return None
    return data if isinstance(data, list) else None


def parse_questions_txt(text: str) -> List[Dict]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    for i, line in enumerate(lines, 1):
        m = re.match(r'^(?:[qQ])?(\d+)[.:\)]\s*(.*)', line)
        qid   = f"q{m.group(1)}" if m else f"q{i}"
        qtext = (m.group(2) if m else line).strip()
        if qtext:
            out.append({"id": qid, "question": qtext})
    return out


def parse_questions_upload(up) -> List[Dict]:
    """Parse questions from an uploaded file object; returns list of {id, question}."""
    if up is None:
        return []
    up.seek(0)
    raw = up.read()
    up.seek(0)
    try:
        if up.name.endswith((".yaml", ".yml")):
            data = yaml.safe_load(raw.decode("utf-8")) or []
        elif up.name.endswith(".json"):
            data = json.loads(raw.decode("utf-8"))
        else:
            return parse_questions_txt(raw.decode("utf-8"))
        return [q for q in (data if isinstance(data, list) else [])
                if isinstance(q, dict) and q.get("question")]
    except Exception:
        return []


def normalize_debug(data: Dict) -> Dict:
    if "metadata" in data:
        return data
    answers = []
    for a in data.get("answers", []):
        answers.append({
            "question_id":   a.get("id", ""),
            "question_type": "?",
            "question":      "",
            "answer":        a.get("answer", ""),
            "status":        "answered",
            "token_usage":   {},
        })
    return {"metadata": {}, "answers": answers}


def _ks_to_store(ks_data: Dict):
    """Build a KnowledgeStore object from the in-memory JSON dict."""
    from src_v4.knowledge_store import KnowledgeStore
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(ks_data, tf, ensure_ascii=False)
        tf_path = tf.name
    try:
        store = KnowledgeStore.load(tf_path)
    finally:
        os.unlink(tf_path)
    return store


# ──────────────────────────── render: demand plan ─────────────────────────────

def render_demand_plan(plan: Dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entity Types",       len(plan.get("entity_types",      [])))
    c2.metric("Numeric Predicates", len(plan.get("numeric_predicates", [])))
    c3.metric("Event Triggers",     len(plan.get("event_triggers",    [])))
    c4.metric("Theme Topics",       len(plan.get("theme_topics",      [])))

    sh("Global Extraction Schema")
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Entity Types", expanded=True):
            for x in plan.get("entity_types", []):
                st.markdown(f"- `{x}`")
        with st.expander("Event Triggers"):
            for x in plan.get("event_triggers", []):
                st.markdown(f"- `{x}`")
    with col_b:
        with st.expander("Numeric Predicates", expanded=True):
            for x in plan.get("numeric_predicates", []):
                st.markdown(f"- `{x}`")
        with st.expander("Theme Topics"):
            for x in plan.get("theme_topics", []):
                st.markdown(f"- `{x}`")

    sh("Per-Question Solver Plans")
    qplans = plan.get("question_plans", {})
    if not qplans:
        st.info("No per-question plans in this file.")
        return
    for qid, qplan in qplans.items():
        solver = qplan.get("solver", "?")
        with st.expander(f"{qid}  ·  [{solver}]"):
            st.json(qplan)


# ──────────────────────────── render: chunks ──────────────────────────────────

def render_chunks(manifest: List[Dict]) -> None:
    sh("Chunk Manifest")
    total_words = sum(c.get("core_word_count", 0) for c in manifest)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Chunks",      len(manifest))
    c2.metric("Total Words",       f"{total_words:,}")
    c3.metric("Avg Words / Chunk", f"{total_words // max(len(manifest), 1):,}")

    rows = [{
        "Chunk":   c.get("chunk_id", ""),
        "Section": c.get("section_heading", ""),
        "Start":   c.get("core_word_start", 0),
        "End":     c.get("core_word_end",   0),
        "Words":   c.get("core_word_count", 0),
    } for c in manifest]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────── render: knowledge store ─────────────────────────

def render_knowledge_store(ks: Dict) -> None:
    entities = ks.get("entities", [])
    facts    = ks.get("facts",    [])
    events   = ks.get("events",   [])
    claims   = ks.get("claims",   [])
    themes   = ks.get("themes",   [])

    sh("Knowledge Store")
    cols = st.columns(5)
    for col, (label, lst) in zip(cols, [
        ("Entities", entities), ("Facts", facts),
        ("Events",   events),   ("Claims", claims), ("Themes", themes),
    ]):
        col.metric(label, len(lst))

    te, tf, tev, tcl, tth = st.tabs(["Entities", "Facts", "Events", "Claims", "Themes"])

    with te:
        if entities:
            rows = []
            for e in entities:
                a = e.get("attributes", {})
                rows.append({
                    "Name":     e.get("name", ""),
                    "Type":     e.get("type", ""),
                    "Country":  a.get("country", a.get("location", "")),
                    "Area km²": a.get("area_km2", ""),
                    "Chunks":   ", ".join(e.get("chunk_ids", [])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No entities.")

    with tf:
        if facts:
            rows = [{
                "Subject":    f.get("subject",         ""),
                "Predicate":  f.get("predicate",        ""),
                "Value":      f.get("value",            ""),
                "Unit":       f.get("unit",             ""),
                "Normalized": f.get("normalized_value", ""),
                "Chunk":      f.get("chunk_id",         ""),
            } for f in facts]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No facts.")

    with tev:
        if events:
            rows = [{
                "Year":        ev.get("year_text",   ""),
                "Year ~":      ev.get("year_approx", ""),
                "Entity":      ev.get("entity",      ""),
                "Description": ev.get("description", ""),
                "Chunk":       ev.get("chunk_id",    ""),
            } for ev in events]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No events.")

    with tcl:
        if claims:
            rows = [{
                "Entity": cl.get("entity",  ""),
                "Claim":  cl.get("text",    ""),
                "Chunk":  cl.get("chunk_id",""),
            } for cl in claims]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No claims.")

    with tth:
        if themes:
            rows = [{
                "Topic":   th.get("topic",   ""),
                "Entity":  th.get("entity",  ""),
                "Context": th.get("context", ""),
                "Chunk":   th.get("chunk_id",""),
            } for th in themes]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No themes.")


# ──────────────────────────── render: answers ─────────────────────────────────

def render_answers(debug: Dict) -> None:
    answers = debug.get("answers", [])
    if not answers:
        st.info("No answers found.")
        return

    sh("Generated Answers")

    # ── question filter selectbox ─────────────────────────────────────────────
    sel_options = ["All questions"] + [
        f"[{ans.get('question_type','?')}]  {ans.get('question_id', ans.get('id',''))}  ·  "
        f"{ans.get('question','')[:60]}{'…' if len(ans.get('question','')) > 60 else ''}"
        for ans in answers
    ]
    selected = st.selectbox("View", sel_options, key="q_filter_sel")

    if selected == "All questions":
        to_show    = answers
        auto_open  = False
    else:
        idx        = sel_options.index(selected) - 1
        to_show    = [answers[idx]]
        auto_open  = True

    # ── render each answer ────────────────────────────────────────────────────
    for ans in to_show:
        qid    = ans.get("question_id", ans.get("id", ""))
        qtype  = ans.get("question_type", "?")
        qtext  = ans.get("question", "")
        answer = ans.get("answer", "")
        tok    = ans.get("token_usage", {})
        pt     = ans.get("processing_time_s")

        # Category visible in collapsed title
        title = (
            f"[{qtype}]  {qid}  ·  "
            f"{qtext[:70]}{'…' if len(qtext) > 70 else ''}"
            if qtext else f"[{qtype}]  {qid}"
        )

        with st.expander(title, expanded=auto_open):
            # Badge + status on one line
            status = ans.get("status", "")
            status_html = (
                f' <span style="color:#64748b;font-size:12px;">{status}</span>'
                if status else ""
            )
            st.markdown(type_badge(qtype) + status_html, unsafe_allow_html=True)

            if qtext:
                st.markdown(f"**Q:** {qtext}")

            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )

            meta_parts = []
            if tok:
                meta_parts.append(
                    f"in {tok.get('input_tokens', 0):,} / out {tok.get('output_tokens', 0):,} tokens"
                )
            if pt is not None:
                meta_parts.append(f"{pt:.2f} s")
            if meta_parts:
                st.caption(" · ".join(meta_parts))


# ──────────────────────────── render: ask a question ─────────────────────────

def render_ask_question(ks_data: Optional[Dict], demand: Optional[Dict]) -> None:
    sh("Ask a Question")

    if not ks_data:
        st.info(
            "Load **knowledge_store.json** first (Demo mode) "
            "or run the pipeline (Live mode) to enable this feature."
        )
        return

    # API key (re-use env or ask)
    api_key = os.environ.get("COURSE_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "COURSE_API_KEY",
            type="password",
            placeholder="Enter your API key",
            key="ask_api_key",
        )

    q_text = st.text_area(
        "Your question",
        placeholder="Ask anything about the document…",
        height=100,
        key="ask_q_text",
    )
    ask_btn = st.button("Get Answer", type="primary", key="ask_btn")

    # Show previous result if available
    if st.session_state.get("s_ask_answer"):
        prev_type   = st.session_state.s_ask_type
        prev_answer = st.session_state.s_ask_answer
        prev_usage  = st.session_state.s_ask_usage or {}
        st.markdown("---")
        st.markdown("**Last answer**")
        st.markdown(type_badge(prev_type), unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-box-ask">{prev_answer}</div>',
            unsafe_allow_html=True,
        )
        if prev_usage:
            st.caption(
                f"in {prev_usage.get('input_tokens', 0):,} / "
                f"out {prev_usage.get('output_tokens', 0):,} tokens"
            )

    if not ask_btn:
        return

    if not q_text.strip():
        st.error("Please enter a question.")
        return
    if not api_key:
        st.error("Please enter your COURSE_API_KEY.")
        return

    os.environ["COURSE_API_KEY"] = api_key

    # Step 1: classify
    qtype = "needle"
    with st.spinner("Classifying question…"):
        try:
            from src_v4.question_classifier import classify_questions_batch
            preds = classify_questions_batch([{"id": "_ask", "question": q_text.strip()}])
            qtype = preds.get("_ask", "needle")
        except Exception as e:
            st.warning(f"Classification failed ({e}), defaulting to 'needle'.")

    st.markdown(f"**Question type:** {type_badge(qtype)}", unsafe_allow_html=True)

    # Step 2: solve
    with st.spinner("Generating answer…"):
        try:
            from src_v4.solvers import solve_question

            store  = _ks_to_store(ks_data)
            dp     = demand or {}
            question = {
                "id":       "_ask",
                "question": q_text.strip(),
                "category": qtype,
            }
            answer, usage = solve_question(question, store, dp)

        except Exception as e:
            st.error(f"Solver error: {e}")
            return

    # Persist result in session state so it survives reruns
    st.session_state.s_ask_type   = qtype
    st.session_state.s_ask_answer = answer
    st.session_state.s_ask_usage  = usage

    st.markdown(
        f'<div class="answer-box-ask">{answer}</div>',
        unsafe_allow_html=True,
    )
    if usage:
        st.caption(
            f"in {usage.get('input_tokens', 0):,} / "
            f"out {usage.get('output_tokens', 0):,} tokens"
        )


# ─────────────────────────────── main UI ──────────────────────────────────────

def main() -> None:
    # ── session state init ────────────────────────────────────────────────────
    for k in ("s_debug", "s_demand", "s_manifest", "s_ks", "s_key",
              "s_ask_type", "s_ask_answer", "s_ask_usage"):
        if k not in st.session_state:
            st.session_state[k] = None

    # ── sidebar ───────────────────────────────────────────────────────────────
    sel_qid: Optional[str] = None  # question selected in Live mode

    with st.sidebar:
        st.markdown("## Distill-and-Solve")
        st.caption("pipeline_v4  ·  Distill → Solve")
        st.divider()

        mode = st.radio(
            "Mode",
            ["Demo · Load outputs", "Live · Run pipeline"],
            index=0,
        )
        st.divider()

        if mode == "Demo · Load outputs":
            st.markdown("**Pipeline outputs**")
            up_debug    = st.file_uploader("answers_debug.json  *(or answers.json)*",
                                           type="json", key="u_debug")
            up_demand   = st.file_uploader("demand_plan.json",
                                           type="json", key="u_demand")
            up_manifest = st.file_uploader("chunk_manifest.json",
                                           type="json", key="u_manifest")
            up_ks       = st.file_uploader("knowledge_store.json",
                                           type="json", key="u_ks")
            st.divider()
            st.markdown("**Answer key** *(optional)*")
            up_key   = st.file_uploader("answer_key  *.yaml / .json*",
                                        type=["yaml", "yml", "json"], key="u_key")
            load_btn = st.button("Load Results", type="primary", use_container_width=True)
            run_btn  = False
            up_doc = up_q = None
            api_key_input = None

        else:  # Live Run
            st.markdown("**API Key**")
            api_key_input = st.text_input(
                "COURSE_API_KEY",
                type="password",
                placeholder="Enter your API key",
                value=os.environ.get("COURSE_API_KEY", ""),
                key="u_api_key",
            )
            st.divider()
            st.markdown("**Document**")
            up_doc = st.file_uploader("Text document  *.txt / .pdf*",
                                      type=["txt", "pdf"], key="u_doc")
            st.markdown("**Questions**")
            up_q   = st.file_uploader("Questions file  *.yaml / .json / .txt*",
                                      type=["yaml", "yml", "json", "txt"], key="u_q")

            # ── question selector ─────────────────────────────────────────────
            q_choices = parse_questions_upload(up_q) if up_q else []
            if q_choices:
                sel_opts = ["All questions"] + [
                    f"{q['id']}  ·  {q['question'][:55]}{'…' if len(q.get('question','')) > 55 else ''}"
                    for q in q_choices
                ]
                sel_label = st.selectbox(
                    "Select question to run",
                    sel_opts,
                    key="u_q_sel",
                )
                if sel_label != "All questions":
                    sel_qid = q_choices[sel_opts.index(sel_label) - 1]["id"]

            st.divider()
            st.markdown("**Answer key** *(optional)*")
            up_key = st.file_uploader("answer_key  *.yaml / .json*",
                                      type=["yaml", "yml", "json"], key="u_key2")
            run_btn  = st.button("Run Pipeline", type="primary", use_container_width=True)
            load_btn = False
            up_debug = up_demand = up_manifest = up_ks = None

        st.divider()
        if st.button("Clear All", use_container_width=True):
            for k in ("s_debug", "s_demand", "s_manifest", "s_ks", "s_key",
                      "s_ask_type", "s_ask_answer", "s_ask_usage"):
                st.session_state[k] = None
            st.rerun()

    # ── page header ───────────────────────────────────────────────────────────
    st.title("Distill-and-Solve Pipeline")
    st.caption(
        "Phase 1 · Demand Planning  →  "
        "Phase 2 · Chunk Distillation  →  "
        "Phase 3 · Question Solving"
    )

    # ── handle load ───────────────────────────────────────────────────────────
    if load_btn:
        loaded = []
        if up_debug:
            d = load_json_upload(up_debug)
            st.session_state.s_debug = normalize_debug(d) if d else None
            if d: loaded.append("answers")
        if up_demand:
            st.session_state.s_demand = load_json_upload(up_demand)
            if st.session_state.s_demand: loaded.append("demand plan")
        if up_manifest:
            st.session_state.s_manifest = load_json_upload(up_manifest)
            if st.session_state.s_manifest: loaded.append("chunk manifest")
        if up_ks:
            st.session_state.s_ks = load_json_upload(up_ks)
            if st.session_state.s_ks: loaded.append("knowledge store")
        if up_key:
            st.session_state.s_key = load_answer_key(up_key)
            if st.session_state.s_key: loaded.append("answer key")
        if loaded:
            st.toast(f"Loaded: {', '.join(loaded)}", icon="✅")

    # ── handle run ────────────────────────────────────────────────────────────
    elif run_btn:
        if not api_key_input:
            st.error("Please enter your COURSE_API_KEY in the sidebar.")
        elif not up_doc or not up_q:
            st.error("Please upload both a document and a questions file.")
        else:
            questions = parse_questions_upload(up_q)
            if not questions:
                st.error("No questions could be parsed from the uploaded file.")
            else:
                doc_bytes = up_doc.read()
                doc_name  = up_doc.name
                clean_qs  = [{"id": q["id"], "question": q["question"]} for q in questions]

                run_label = (
                    f"question {sel_qid}" if sel_qid
                    else f"{len(questions)} questions"
                )

                success = False
                with tempfile.TemporaryDirectory() as td:
                    doc_path = os.path.join(td, doc_name)
                    q_path   = os.path.join(td, "questions.yaml")
                    with open(doc_path, "wb") as f:
                        f.write(doc_bytes)
                    with open(q_path, "w", encoding="utf-8") as f:
                        yaml.dump(clean_qs, f, allow_unicode=True)

                    with st.spinner(f"Running pipeline on {run_label}…"):
                        try:
                            os.environ["COURSE_API_KEY"] = api_key_input
                            from src_v4.pipeline import run_pipeline
                            run_pipeline(
                                document_path=doc_path,
                                yaml_path=q_path,
                                question_ids=[sel_qid] if sel_qid else None,
                                verbose=False,
                                debug=True,
                            )
                            success = True
                        except Exception as e:
                            st.error(f"Pipeline error: {e}")

                if success:
                    out = PIPELINE_DIR / "outputs_v4"
                    for fname, sk in [
                        ("answers_debug.json",  "s_debug"),
                        ("demand_plan.json",    "s_demand"),
                        ("chunk_manifest.json", "s_manifest"),
                        ("knowledge_store.json","s_ks"),
                    ]:
                        fp = out / fname
                        if fp.exists():
                            with open(fp, encoding="utf-8") as f:
                                data = json.load(f)
                            if sk == "s_debug":
                                data = normalize_debug(data)
                            st.session_state[sk] = data
                    if up_key:
                        st.session_state.s_key = load_answer_key(up_key)
                    st.success(f"Pipeline complete ({run_label}).")

    # ── display ───────────────────────────────────────────────────────────────
    debug    = st.session_state.s_debug
    demand   = st.session_state.s_demand
    manifest = st.session_state.s_manifest
    ks       = st.session_state.s_ks

    if not any([debug, demand, manifest, ks]):
        _render_welcome()
        return

    # metadata bar
    meta       = (debug or {}).get("metadata", {})
    n_q        = meta.get("total_questions", len((debug or {}).get("answers", [])))
    n_ch       = meta.get("total_chunks", len(manifest) if manifest else "–")
    tu         = meta.get("total_usage", {})
    model_name = meta.get("model", "–")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Questions",     n_q)
    c2.metric("Chunks",        n_ch)
    c3.metric("Model",         model_name)
    c4.metric("Input Tokens",  f"{tu.get('input_tokens',  0):,}" if tu else "–")
    c5.metric("Output Tokens", f"{tu.get('output_tokens', 0):,}" if tu else "–")
    st.divider()

    # tabs
    tab_names = [
        "Phase 1 · Demand Plan",
        "Phase 2 · Chunks",
        "Phase 2 · Knowledge Store",
        "Phase 3 · Answers",
        "Ask a Question",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if demand:
            render_demand_plan(demand)
        else:
            st.info("Upload **demand_plan.json** in the sidebar to view Phase 1.")

    with tabs[1]:
        if manifest:
            render_chunks(manifest)
        else:
            st.info("Upload **chunk_manifest.json** in the sidebar to view chunking.")

    with tabs[2]:
        if ks:
            render_knowledge_store(ks)
        else:
            st.info("Upload **knowledge_store.json** in the sidebar to view the Knowledge Store.")

    with tabs[3]:
        if debug:
            render_answers(debug)
        else:
            st.info("Upload **answers_debug.json** (or answers.json) to view answers.")

    with tabs[4]:
        render_ask_question(ks, demand)


# ── welcome screen ────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.info(
        "Upload pipeline output files in the sidebar and click **Load Results**, "
        "or switch to **Live · Run pipeline** to execute the pipeline from scratch."
    )
    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            '<div class="welcome-phase">'
            '<h4>Phase 1 · Demand Plan</h4>'
            '<p>One LLM call analyzes all questions and produces a <strong>global extraction schema</strong>: '
            'which entity types, numeric predicates, event triggers, and theme topics to look for '
            'across every chunk of the document.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="welcome-phase">'
            '<h4>Phase 2 · Chunk Distillation</h4>'
            '<p>The document is split into chunks (section-aware or word-count based with 150-word overlap). '
            'Each chunk receives <strong>one LLM call</strong> that extracts structured JSON: '
            'entities, facts, events, claims, themes. '
            'Results are merged into a <strong>Knowledge Store</strong>.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            '<div class="welcome-phase">'
            '<h4>Phase 3 · Question Solving</h4>'
            '<p>Each question is classified into one of <strong>7 types</strong> and solved with a '
            'type-specific strategy. Python-first logic handles aggregation, superlative, absence, '
            'and contradiction; LLM synthesis handles needle, cross-section, and global synthesis.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Question types supported:**")
    st.markdown(
        "  ".join(
            f'<span class="badge" style="background:{c};color:#fff;">{t}</span>'
            for t, c in TYPE_COLORS.items()
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    with st.expander("Expected file formats"):
        st.markdown("""
**answers_debug.json** — full debug output with question types, token usage, timing
**answers.json** — submission format `{"team": "...", "answers": [{"id": "...", "answer": "..."}]}`
**demand_plan.json** — Phase 1 output: entity_types, numeric_predicates, question_plans, …
**chunk_manifest.json** — list of chunks with section headings and word ranges
**knowledge_store.json** — merged entities / facts / events / claims / themes

**Questions file** (.yaml, .json, or .txt):
```yaml
- id: q1
  question: "How many national parks does the book profile in total?"
```
Or plain text:
```
1. How many national parks does the book profile?
2. Which park covers the largest area?
```
""")


if __name__ == "__main__":
    main()

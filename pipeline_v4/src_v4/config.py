import os
"""
Pipeline configuration — Distill → Solve architecture (pipeline_v4).
"""

# ── Course broker ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("COURSE_API_BASE_URL", "")
API_KEY      = None   # None = read from env var at runtime

# ── Models ────────────────────────────────────────────────────────────────────
MAIN_MODEL  = "mistral-small-3-2"
SMALL_MODEL = "ministral-3b-2512"
VALID_MODELS = {MAIN_MODEL, SMALL_MODEL}

PLANNER_MODEL   = MAIN_MODEL
DISTILLER_MODEL = MAIN_MODEL
SOLVER_MODEL    = MAIN_MODEL

# ── Generation parameters ─────────────────────────────────────────────────────
TEMPERATURE_FACTUAL = 0.1
TEMPERATURE_ANSWER  = 0.3
TOP_P               = 0.95
SEED                = 42

MAX_TOKENS_PLANNER   = 4000
MAX_TOKENS_DISTILLER = 16000
MAX_TOKENS_SOLVER    = 2000

# ── Document chunking ─────────────────────────────────────────────────────────
CORE_CHUNK_WORDS     = 30000
CONTEXT_WORDS_BEFORE = 100
CONTEXT_WORDS_AFTER  = 100

# ── Section chunking ─────────────────────────────────────────────────────────
USE_SECTION_CHUNKER   = True
SECTION_MIN_WORDS     = 300
SECTION_MAX_WORDS     = 10000
SECTION_CONTEXT_WORDS = 150

# ── Concurrency ───────────────────────────────────────────────────────────────
DISTILLATION_CONCURRENCY = 4

# ── Retry / resilience ────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 3

# ── Token budget ─────────────────────────────────────────────────────────────
TOKEN_BUDGET_INPUT = 64_000

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR          = "outputs_v4"
ANSWERS_FILE        = "outputs_v4/answers.json"
DEBUG_FILE          = "outputs_v4/answers_debug.json"
DEMAND_PLAN_FILE    = "outputs_v4/demand_plan.json"
CHUNK_MANIFEST_FILE = "outputs_v4/chunk_manifest.json"
CHUNK_DISTILLS_DIR  = "outputs_v4/chunk_distills"
STORE_FILE          = "outputs_v4/knowledge_store.json"
TOKEN_USAGE_FILE    = "outputs_v4/token_usage.json"

# ── Submission metadata ───────────────────────────────────────────────────────
TEAM_NAME      = "JJaeseo"
PIPELINE_NOTES = (
    "Distill-and-Solve: Phase 1 builds a demand plan from questions. "
    "Phase 2 distills every chunk into structured JSON (entities/facts/events/themes) "
    "with one LLM call per chunk. Phase 3 solves questions using Python-first logic "
    "(aggregation=count, superlative=max/min, absence=set-check, contradiction=diff-detect) "
    "with LLM synthesis only for needle/cross_section/global_synthesis. "
    "No RAG, no embeddings, no chunk selection — all chunks processed in order."
)

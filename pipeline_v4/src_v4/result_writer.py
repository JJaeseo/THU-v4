"""
Save pipeline results to JSON files.
  answers.json       — submission format
  answers_debug.json — full detail with token usage
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import OUTPUT_DIR, ANSWERS_FILE, DEBUG_FILE, DISTILLER_MODEL, TEAM_NAME, PIPELINE_NOTES

logger = logging.getLogger(__name__)


def save_results(
    results:       List[Dict[str, Any]],
    document_path: str,
    n_chunks:      int                      = 0,
    total_usage:   Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    answers_list = [
        {"id": r.get("question_id", "unknown"), "answer": r.get("answer", "")}
        for r in results
    ]

    submission = {"team": TEAM_NAME, "notes": PIPELINE_NOTES, "answers": answers_list}
    _write_verified(ANSWERS_FILE, submission)

    debug: Dict[str, Any] = {
        "metadata": {
            "method":          "distill_and_solve",
            "model":           DISTILLER_MODEL,
            "document":        str(document_path),
            "total_questions": len(results),
            "total_chunks":    n_chunks,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "total_usage":     total_usage or {},
        },
        "answers": [
            {
                "question_id":       r.get("question_id"),
                "question_type":     r.get("question_type"),
                "question":          r.get("question"),
                "answer":            r.get("answer"),
                "status":            r.get("status", "answered"),
                "token_usage":       r.get("token_usage", {}),
                "processing_time_s": r.get("processing_time_s"),
            }
            for r in results
        ],
    }
    _write_verified(DEBUG_FILE, debug)

    logger.info("Submission : %s", ANSWERS_FILE)
    logger.info("Debug      : %s", DEBUG_FILE)
    return {"submission": ANSWERS_FILE, "debug": DEBUG_FILE}


def _write_verified(path: str, data: Any) -> None:
    serialised = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if "NaN" in serialised or "Infinity" in serialised:
        raise ValueError(f"JSON contains NaN or Infinity — cannot write {path}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(serialised)
    with open(path, encoding="utf-8") as fh:
        json.load(fh)
    logger.info("Saved and verified: %s (%d bytes)", path, len(serialised))

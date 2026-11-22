"""
Load and validate questions from a YAML file.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def load_questions(yaml_path: str) -> List[Dict[str, Any]]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Question YAML not found: {yaml_path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, list):
        raise ValueError("Question YAML must be a list of question objects.")

    seen_ids: set = set()
    questions:    List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Each question must be a dict; got {type(item).__name__}")
        qid = item.get("id")
        if qid is None:
            raise ValueError(f"Question missing 'id' field: {item}")
        if qid in seen_ids:
            raise ValueError(f"Duplicate question id: {qid!r}")
        seen_ids.add(qid)

        q_text = item.get("question", "")
        if isinstance(q_text, str):
            item["question"] = q_text.strip()

        questions.append(item)

    logger.info("Loaded %d questions from %s", len(questions), yaml_path)
    return questions

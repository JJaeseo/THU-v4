"""
Document Inventory Scanner — pure regex, no LLM calls.

Scans the full document text once for all labeled structural elements
(figures, boxes, main tables, annex tables) and returns exact per-chapter
counts.  Results are stored in KnowledgeStore._inventory and used by
the aggregation solver when counting questions target these elements.
"""

import re
from typing import Any, Dict, List

# ── Regex patterns ────────────────────────────────────────────────────────────
# Annex tables: "Annex Table 5.A.1"  (chapter.LETTER.number)
_RE_ANNEX = re.compile(
    r'\bAnnex\s+Table\s+(\d+\.[A-Z]\.\d+)\b',
    re.IGNORECASE,
)

# Main figures: "Figure 1.3"  (chapter.number — no letter sub-chapter)
_RE_FIGURE = re.compile(r'\bFigure\s+(\d+\.\d+)\b', re.IGNORECASE)

# Main boxes: "Box 1.2"
_RE_BOX = re.compile(r'\bBox\s+(\d+\.\d+)\b', re.IGNORECASE)

# Main tables: match only DEFINITIONS — "Table 5.1. Title" (period + text after label).
# Cross-references like "as shown in Table 6.11 for…" have no trailing period,
# so they are excluded.  Annex labels ("5.A.1") contain letters so they never
# match \d+\.\d+ regardless.
_RE_TABLE = re.compile(r'\bTable\s+(\d+\.\d+)\.', re.IGNORECASE)


def scan_document_inventory(text: str) -> Dict[str, Any]:
    """
    Scan *text* for all unique Figure / Box / Table labels.

    Returns a dict:
      {
        "figures":      {chapter_str: [sorted label list]},
        "boxes":        {chapter_str: [sorted label list]},
        "main_tables":  {chapter_str: [sorted label list]},
        "annex_tables": {chapter_str: [sorted label list]},
      }

    All label sets are de-duplicated: a label mentioned 20 times is counted once.
    """
    figures:      set = set()
    boxes:        set = set()
    main_tables:  set = set()
    annex_tables: set = set()

    for m in _RE_ANNEX.finditer(text):
        annex_tables.add(m.group(1).upper())

    for m in _RE_FIGURE.finditer(text):
        figures.add(m.group(1))

    for m in _RE_BOX.finditer(text):
        boxes.add(m.group(1))

    for m in _RE_TABLE.finditer(text):
        main_tables.add(m.group(1))

    return {
        "figures":      _by_chapter(figures),
        "boxes":        _by_chapter(boxes),
        "main_tables":  _by_chapter(main_tables),
        "annex_tables": _by_chapter(annex_tables),
    }


def _by_chapter(labels: set) -> Dict[str, List[str]]:
    """Group labels by their chapter prefix and sort within each chapter."""
    d: Dict[str, List[str]] = {}
    for lbl in labels:
        ch = lbl.split('.')[0]
        d.setdefault(ch, []).append(lbl)

    def _sort_key(x: str) -> List:
        return [int(p) if p.isdigit() else p for p in x.replace('.', ' ').split()]

    for ch in d:
        d[ch] = sorted(d[ch], key=_sort_key)

    return dict(sorted(d.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999))


def inventory_summary(inventory: Dict[str, Any]) -> str:
    """One-line summary for log output."""
    def _total(key: str) -> int:
        return sum(len(v) for v in inventory.get(key, {}).values())

    return (
        f"figures={_total('figures')}  boxes={_total('boxes')}  "
        f"main_tables={_total('main_tables')}  annex_tables={_total('annex_tables')}"
    )

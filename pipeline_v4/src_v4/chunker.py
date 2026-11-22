"""
Word-offset document chunker with ±context windows.

Each chunk = previous_context + core_content + next_context.
Core content is non-overlapping; context exists for boundary safety.
Table blocks (detected by column-aligned numeric lines) are kept atomic —
a chunk boundary is never placed inside a table block.
"""

import re
from typing import Any, Dict, List, Tuple

from .config import CORE_CHUNK_WORDS, CONTEXT_WORDS_BEFORE, CONTEXT_WORDS_AFTER

# ── Table detection (mirrors universal_chunker, kept local to avoid coupling) ─
_RE_TABLE_LABEL = re.compile(
    r'^\s*(?:Table|Annex\s+Table|Figure|Box)\s+\d', re.IGNORECASE
)
_RE_PIPE_ROW  = re.compile(r'^\s*\|')
_RE_COL_DATA  = re.compile(r'\d[\d,.\s]*\s{2,}\d')
_RE_SEPARATOR = re.compile(r'^[\-\|:+\s]{4,}$')


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return bool(s and (
        _RE_TABLE_LABEL.match(s) or
        _RE_PIPE_ROW.match(s) or
        _RE_SEPARATOR.match(s) or
        _RE_COL_DATA.search(s)
    ))


def _compute_table_word_ranges(text: str) -> List[Tuple[int, int]]:
    """
    Return (word_start, word_end) word-index ranges for table-like line groups.
    A table block starts at the first table line and ends at the last
    consecutive table line (blank lines break the block).
    """
    ranges: List[Tuple[int, int]] = []
    word_pos    = 0
    in_table    = False
    tbl_start   = 0
    blank_words = 0   # words accumulated on blank lines while inside table

    for line in text.split('\n'):
        lw = len(line.split())
        if not line.strip():
            if in_table:
                blank_words += lw  # blank line may separate header from rows
            else:
                word_pos += lw
            continue

        if _is_table_line(line):
            if not in_table:
                in_table  = True
                tbl_start = word_pos
                blank_words = 0
            word_pos += blank_words + lw
            blank_words = 0
        else:
            if in_table:
                ranges.append((tbl_start, word_pos))
                in_table  = False
                word_pos += blank_words + lw
                blank_words = 0
            else:
                word_pos += lw

    if in_table:
        ranges.append((tbl_start, word_pos))
    return ranges


def _snap_avoiding_tables(
    pos:          int,
    table_ranges: List[Tuple[int, int]],
    limit:        int,
) -> int:
    """
    If *pos* falls inside a table range, advance to the end of that range.
    Falls back to the range start if advancing would exceed *limit*.
    """
    for t_start, t_end in table_ranges:
        if t_start < pos < t_end:
            if t_end <= limit:
                return t_end
            return t_start if t_start > 0 else pos
    return pos


def build_chunks(
    text:               str,
    core_chunk_words:   int = None,
    context_before:     int = None,
    context_after:      int = None,
) -> List[Dict[str, Any]]:
    csz     = core_chunk_words or CORE_CHUNK_WORDS
    ctx_bef = context_before   if context_before is not None else CONTEXT_WORDS_BEFORE
    ctx_aft = context_after    if context_after  is not None else CONTEXT_WORDS_AFTER

    words        = text.split()
    n            = len(words)
    if n == 0:
        return []

    table_ranges = _compute_table_word_ranges(text)

    chunks: List[Dict[str, Any]] = []
    idx      = 0
    core_pos = 0

    while core_pos < n:
        core_end = min(core_pos + csz, n)
        core_end = _snap_to_sentence(words, core_end, window=50, limit=n)
        core_end = _snap_avoiding_tables(core_end, table_ranges, n)

        prev_start = max(0,   core_pos - ctx_bef)
        next_end   = min(n,   core_end + ctx_aft)

        prev_text = " ".join(words[prev_start:core_pos])
        core_text = " ".join(words[core_pos:core_end])
        next_text = " ".join(words[core_end:next_end])

        parts     = [p for p in (prev_text, core_text, next_text) if p]
        full_text = "\n\n".join(parts)

        chunks.append({
            "chunk_id":           f"chunk_{idx:03d}",
            "chunk_index":        idx,
            "core_word_start":    core_pos,
            "core_word_end":      core_end,
            "context_word_start": prev_start,
            "context_word_end":   next_end,
            "word_count":         len(full_text.split()),
            "core_word_count":    len(words[core_pos:core_end]),
            "text":               full_text,
            "previous_context":   prev_text,
            "core_content":       core_text,
            "next_context":       next_text,
        })

        idx      += 1
        core_pos  = core_end

    return chunks


def validate_coverage(chunks: List[Dict[str, Any]], total_words: int) -> bool:
    import logging
    log = logging.getLogger(__name__)

    if not chunks:
        log.warning("No chunks produced.")
        return False

    expected_pos = 0
    ok = True
    for c in chunks:
        if c["core_word_start"] != expected_pos:
            log.warning(
                "Coverage gap: expected %d, got %d in %s",
                expected_pos, c["core_word_start"], c["chunk_id"],
            )
            ok = False
        expected_pos = c["core_word_end"]

    if expected_pos != total_words:
        log.warning(
            "Coverage incomplete: last core_word_end=%d but total_words=%d",
            expected_pos, total_words,
        )
        ok = False
    return ok


def _snap_to_sentence(words: List[str], pos: int, window: int, limit: int) -> int:
    search_end = min(pos + window, limit)
    for i in range(pos, search_end):
        w = words[i]
        if w and w[-1] in '.!?':
            return i + 1
    return pos

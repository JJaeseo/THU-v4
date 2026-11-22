"""
Universal structure-aware document chunker.

Structure detection priority (first match with >= 3 sections wins):
  1. H1 Markdown  (# Title)
  2. H2 Markdown  (## Title)
  3. H3 Markdown  (### Title)
  4. Chapter/Section/Part
  5. Decimal sections  (1. Title)
  6. ALL CAPS headings

Falls back to fixed word-count chunking if no structure detected.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MIN_PROBE_WORDS   = 80
_BOUNDARY_MIN_WORDS = 200


def build_section_chunks(
    text:              str,
    min_section_words: int = 300,
    max_section_words: int = 10_000,
    context_words:     int = 200,
) -> List[Dict[str, Any]]:
    lines = text.split('\n')
    style, boundaries = _detect_structure(lines)

    if style is None or len(boundaries) < 3:
        logger.info("No heading structure detected — using fixed word-count chunker (chunk_size=%d).", max_section_words)
        print(f"[DEBUG] _detect_structure: style={style}, boundaries={len(boundaries)} → fallback to build_chunks(chunk_size={max_section_words})", flush=True)
        from .chunker import build_chunks
        return build_chunks(text, core_chunk_words=max_section_words)

    logger.info("Universal chunker: style=%s, %d boundaries.", style, len(boundaries))

    all_boundaries = sorted(set([0] + boundaries + [len(lines)]))
    raw_sections   = _build_raw_sections(lines, all_boundaries)
    sized          = _merge_and_split(raw_sections, min_section_words, max_section_words)

    logger.info("Final chunks: %d  (min=%d max=%d words each)", len(sized), min_section_words, max_section_words)
    return _build_chunk_dicts(sized, context_words)


def validate_section_coverage(chunks: List[Dict[str, Any]], total_words: int) -> bool:
    if not chunks:
        logger.warning("No section chunks produced.")
        return False
    expected, ok = 0, True
    for c in chunks:
        if c["core_word_start"] != expected:
            logger.warning("Coverage gap at %s: expected %d, got %d",
                           c["chunk_id"], expected, c["core_word_start"])
            ok = False
        expected = c["core_word_end"]
    if expected != total_words:
        logger.warning("Coverage incomplete: last=%d, total=%d", expected, total_words)
        ok = False
    return ok


_HEADING_CANDIDATES: List[Tuple[str, re.Pattern]] = [
    ("h1",      re.compile(r'^# [^#\s]')),
    ("h2",      re.compile(r'^## [^#\s]')),
    ("h3",      re.compile(r'^### [^#\s]')),
    ("chapter", re.compile(r'^(Chapter|Section|Part)\s+[\dIVXivx]+', re.I)),
    ("decimal", re.compile(r'^\d+\.\s+[A-ZÀ-ž]')),
]


def _detect_structure(lines: List[str]) -> Tuple[Optional[str], List[int]]:
    n_lines = len(lines)
    for style, pattern in _HEADING_CANDIDATES:
        raw = [i for i, l in enumerate(lines) if pattern.match(l.strip())]
        if len(raw) < 3:
            continue
        if _median_word_count(lines, raw) < _MIN_PROBE_WORDS:
            continue
        filtered = _filter_boundaries_by_size(lines, raw, n_lines)
        if len(filtered) >= 3:
            logger.info("Boundary filter: style=%s  raw=%d → qualified=%d", style, len(raw), len(filtered))
            return style, filtered

    caps = [i for i, l in enumerate(lines) if _is_caps_heading(l.strip())]
    if len(caps) >= 3 and _median_word_count(lines, caps) >= _MIN_PROBE_WORDS:
        filtered = _filter_boundaries_by_size(lines, caps, n_lines)
        if len(filtered) >= 3:
            return "allcaps", filtered

    return None, []


def _filter_boundaries_by_size(lines: List[str], boundaries: List[int], n_lines: int) -> List[int]:
    kept = []
    for j, b in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else n_lines
        if len(" ".join(lines[b:end]).split()) >= _BOUNDARY_MIN_WORDS:
            kept.append(b)
    return kept


def _median_word_count(lines: List[str], boundaries: List[int]) -> int:
    counts = []
    for j, b in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(lines)
        counts.append(len(' '.join(lines[b:end]).split()))
    return sorted(counts)[len(counts) // 2] if counts else 0


def _is_caps_heading(s: str) -> bool:
    if not s or s[0] in ('#', '-', '*', '>', '<', '|', '!', '['):
        return False
    if len(s) < 5 or len(s) > 100:
        return False
    words = s.split()
    if len(words) < 2:
        return False
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.8


# ── Table-aware splitting helpers ─────────────────────────────────────────────

_RE_TABLE_LABEL = re.compile(
    r'^\s*(?:Table|Annex\s+Table|Figure|Box)\s+\d', re.IGNORECASE
)
_RE_PIPE_ROW  = re.compile(r'^\s*\|')
_RE_COL_DATA  = re.compile(r'\d[\d,.\s]*\s{2,}\d')   # num  [2+ spaces]  num
_RE_SEPARATOR = re.compile(r'^[\-\|:+\s]{4,}$')


def _is_table_line(line: str) -> bool:
    """
    True when *line* looks like part of a table:
    a table/figure/box label, a pipe row, a separator, or two numbers
    separated by ≥ 2 spaces (column alignment).
    Conservative enough to avoid matching normal prose.
    """
    s = line.strip()
    return bool(s and (
        _RE_TABLE_LABEL.match(s) or
        _RE_PIPE_ROW.match(s) or
        _RE_SEPARATOR.match(s) or
        _RE_COL_DATA.search(s)
    ))


def _segment_by_tables(text: str) -> List[Tuple[str, bool]]:
    """
    Partition *text* into (block_text, is_table_block) segments.

    A paragraph is a table block when ≥ 50 % of its non-empty lines pass
    _is_table_line().  Adjacent segments of the same type are merged so the
    table header paragraph and its data rows are always kept together.
    """
    paras = [p.strip() for p in re.split(r'\n{2,}', text.strip()) if p.strip()]
    raw: List[Tuple[str, bool]] = []

    for para in paras:
        lines = [l for l in para.split('\n') if l.strip()]
        if not lines:
            continue
        tbl_count = sum(1 for l in lines if _is_table_line(l))
        is_tbl = len(lines) >= 2 and (tbl_count / len(lines)) >= 0.5
        raw.append((para, is_tbl))

    # Merge consecutive segments of the same type
    merged: List[List] = []
    for para, is_tbl in raw:
        if merged and merged[-1][1] == is_tbl:
            merged[-1][0] += '\n\n' + para
        else:
            merged.append([para, is_tbl])

    return [(s, t) for s, t in merged]


def _split_at_rows(table_text: str, max_words: int) -> List[str]:
    """
    Split an oversized table at line (row) boundaries instead of mid-row.
    Preserves the header row at the start of every subsequent chunk.
    """
    lines = table_text.split('\n')
    # Detect header: first non-empty line(s) before any data row
    header_lines: List[str] = []
    for l in lines:
        if l.strip():
            header_lines.append(l)
            if not _is_table_line(l) or _RE_TABLE_LABEL.match(l.strip()):
                break  # title line only; stop
        if len(header_lines) >= 2:
            break

    header_words = sum(len(l.split()) for l in header_lines)
    chunks: List[str] = []
    current: List[str] = list(header_lines)
    current_words = header_words
    past_header = len(header_lines)

    for line in lines[past_header:]:
        lw = len(line.split())
        if current_words + lw > max_words and len(current) > len(header_lines):
            chunks.append('\n'.join(current))
            current = list(header_lines)
            current_words = header_words
        current.append(line)
        current_words += lw

    if current:
        chunks.append('\n'.join(current))
    return chunks or [table_text]


def _pack_with_table_awareness(
    segments:  List[Tuple[str, bool]],
    max_words: int,
) -> List[str]:
    """
    Pack (text, is_table) segments into max_words-sized chunks.
    Table blocks are atomic — never split mid-block.
    Oversized table blocks are split at row boundaries.
    Oversized prose blocks are split at sentence boundaries.
    """
    result:  List[str] = []
    pending: List[str] = []
    pending_words = 0

    for content, is_table in segments:
        unit_words = len(content.split())

        if pending_words + unit_words > max_words and pending:
            result.append('\n\n'.join(pending))
            pending, pending_words = [], 0

        if is_table and unit_words > max_words:
            if pending:
                result.append('\n\n'.join(pending))
                pending, pending_words = [], 0
            result.extend(_split_at_rows(content, max_words))
        elif not is_table and unit_words > max_words:
            # Oversized prose block (e.g. PDF with sparse paragraph breaks)
            # — split at sentence boundaries before packing
            if pending:
                result.append('\n\n'.join(pending))
                pending, pending_words = [], 0
            sentences = re.split(r'(?<=[.!?])\s+', content.strip())
            result.extend(_pack(sentences, max_words))
        else:
            pending.append(content)
            pending_words += unit_words

    if pending:
        result.append('\n\n'.join(pending))
    return result or [c for c, _ in segments]


def _build_raw_sections(lines: List[str], boundaries: List[int]) -> List[Dict[str, Any]]:
    sections = []
    for i in range(len(boundaries) - 1):
        start, end  = boundaries[i], boundaries[i + 1]
        chunk_lines = lines[start:end]
        body_lines  = [
            l.strip() for l in chunk_lines
            if l.strip() and not (l.strip().startswith('<!--') and l.strip().endswith('-->'))
        ]
        body = '\n'.join(body_lines)
        wc   = len(body.split())
        if wc < 5:
            continue
        heading = _extract_heading(chunk_lines)
        sections.append({"heading": heading, "body": body, "word_count": wc})
    return sections


def _extract_heading(chunk_lines: List[str]) -> str:
    for l in chunk_lines[:8]:
        s = l.strip()
        if not s or s.startswith('<!--'):
            continue
        if s.startswith('#'):
            return re.sub(r'^#+\s*', '', s).strip()
        return s
    return ""


def _merge_and_split(
    sections:  List[Dict[str, Any]],
    min_words: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    split: List[Dict[str, Any]] = []
    for sec in sections:
        print(f"[DEBUG] section '{sec['heading'][:40]}': {sec['word_count']} words", flush=True)
        if sec["word_count"] <= max_words:
            split.append(sec)
        else:
            subs = _split_text(sec["body"], max_words)
            print(f"[DEBUG]   → _split_text produced {len(subs)} sub-chunk(s)", flush=True)
            for idx, sub in enumerate(subs):
                h = sec["heading"] if idx == 0 else (f"{sec['heading']} (cont.)" if sec["heading"] else "")
                split.append({"heading": h, "body": sub, "word_count": len(sub.split())})

    merged: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None

    for sec in split:
        if pending is None:
            pending = sec
            continue
        if pending["word_count"] < min_words:
            combined = (pending["body"] + "\n\n" + sec["body"]).strip()
            pending  = {
                "heading":    pending["heading"] or sec["heading"],
                "body":       combined,
                "word_count": len(combined.split()),
            }
        else:
            merged.append(pending)
            pending = sec

    if pending:
        merged.append(pending)
    return merged


def _split_text(text: str, max_words: int) -> List[str]:
    segments = _segment_by_tables(text)
    if any(is_tbl for _, is_tbl in segments):
        # Table blocks detected — use table-aware packing
        return _pack_with_table_awareness(segments, max_words)
    # No tables: original paragraph → sentence fallback
    paragraphs = re.split(r'\n{2,}', text.strip())
    if len(paragraphs) > 1:
        return _pack(paragraphs, max_words)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return _pack(sentences, max_words)


def _pack(units: List[str], max_words: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for unit in units:
        uw = len(unit.split())
        if current_words + uw > max_words and current:
            chunks.append('\n\n'.join(current))
            current, current_words = [], 0
        current.append(unit)
        current_words += uw
    if current:
        chunks.append('\n\n'.join(current))
    return chunks or ['\n\n'.join(units)]


def _build_chunk_dicts(sections: List[Dict[str, Any]], context_words: int) -> List[Dict[str, Any]]:
    all_words: List[str] = []
    section_ranges: List[Tuple[int, int]] = []

    for sec in sections:
        body_words = sec["body"].split()
        start      = len(all_words)
        all_words.extend(body_words)
        section_ranges.append((start, len(all_words)))

    total  = len(all_words)
    chunks = []

    for idx, (sec, (core_s, core_e)) in enumerate(zip(sections, section_ranges)):
        prev_start = max(0,     core_s - context_words)
        next_end   = min(total, core_e + context_words)

        prev_text = " ".join(all_words[prev_start:core_s])
        core_text = " ".join(all_words[core_s:core_e])
        next_text = " ".join(all_words[core_e:next_end])

        parts     = [p for p in (prev_text, core_text, next_text) if p]
        full_text = "\n\n".join(parts)

        chunks.append({
            "chunk_id":           f"chunk_{idx:03d}",
            "chunk_index":        idx,
            "core_word_start":    core_s,
            "core_word_end":      core_e,
            "context_word_start": prev_start,
            "context_word_end":   next_end,
            "word_count":         len(full_text.split()),
            "core_word_count":    len(all_words[core_s:core_e]),
            "text":               full_text,
            "previous_context":   prev_text,
            "core_content":       core_text,
            "next_context":       next_text,
            "section_heading":    sec.get("heading", ""),
        })

    return chunks

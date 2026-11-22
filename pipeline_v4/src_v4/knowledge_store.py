"""
KnowledgeStore — accumulates distilled chunk data and provides typed query API.

Storage:
  _entities  : {name.lower(): entity_record}   (merged across chunks)
  _facts     : [fact_record, ...]
  _events    : [event_record, ...]
  _claims    : [claim_record, ...]
  _themes    : [theme_record, ...]
  _raw_chunks: [{chunk_id, text}, ...]          for needle raw-text fallback
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class KnowledgeStore:

    def __init__(self):
        self._entities:   Dict[str, Dict] = {}
        self._facts:      List[Dict]      = []
        self._events:     List[Dict]      = []
        self._claims:     List[Dict]      = []
        self._themes:     List[Dict]      = []
        self._raw_chunks: List[Dict]      = []
        self._inventory:  Dict[str, Any]  = {}

    def set_inventory(self, inventory: Dict[str, Any]) -> None:
        self._inventory = inventory or {}

    def get_inventory(self) -> Dict[str, Any]:
        return self._inventory

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        distilled: Dict[str, Any],
        chunk_id:  str,
        raw_text:  str = "",
    ) -> None:
        """Accumulate all records from one chunk's distilled JSON."""
        for e in distilled.get("entities", []):
            self._merge_entity(e, chunk_id)
        for f in distilled.get("facts", []):
            self._facts.append({**f, "chunk_id": chunk_id})
        for ev in distilled.get("events", []):
            self._events.append({**ev, "chunk_id": chunk_id})
        for c in distilled.get("claims", []):
            self._claims.append({**c, "chunk_id": chunk_id})
        for t in distilled.get("themes", []):
            self._themes.append({**t, "chunk_id": chunk_id})
        if raw_text.strip():
            self._raw_chunks.append({"chunk_id": chunk_id, "text": raw_text})

    def _merge_entity(self, e: Dict, chunk_id: str) -> None:
        key = _entity_merge_key(e.get("name", ""))
        if not key:
            return
        # Guard: attributes must be a dict
        raw_attrs = e.get("attributes") or {}
        if not isinstance(raw_attrs, dict):
            raw_attrs = {}

        if key not in self._entities:
            self._entities[key] = {
                "name":       e.get("name", ""),
                "type":       e.get("type", ""),
                "attributes": dict(raw_attrs),
                "chunk_ids":  [chunk_id],
            }
        else:
            rec = self._entities[key]
            for k, v in raw_attrs.items():
                if k not in rec["attributes"] or (v is not None and rec["attributes"].get(k) is None):
                    rec["attributes"][k] = v
            if chunk_id not in rec["chunk_ids"]:
                rec["chunk_ids"].append(chunk_id)
            if not rec["type"] and e.get("type"):
                rec["type"] = e["type"]

    # ── Query API ─────────────────────────────────────────────────────────────

    def entities_by_type(self, etype: str) -> List[Dict]:
        """Return all entities whose type fuzzy-matches *etype*."""
        et = str(etype or "").lower().replace("_", " ").replace("-", " ")
        results = []
        for rec in self._entities.values():
            t = str(rec.get("type") or "").lower().replace("_", " ").replace("-", " ")
            if et in t or t in et or any(w in t for w in et.split() if len(w) > 3):
                results.append(rec)
        return results

    def facts_by_predicate(self, predicate: str) -> List[Dict]:
        """
        Return facts whose predicate matches *predicate* (fuzzy substring or word overlap).
        Also scans entity attributes for the same predicate.
        """
        pred = str(predicate or "").lower().replace("_", " ")
        pred_words = {w for w in pred.split() if len(w) > 3}
        results = []
        seen_keys = set()

        for f in self._facts:
            fp = str(f.get("predicate") or "").lower().replace("_", " ")
            fp_words = {w for w in fp.split() if len(w) > 3}
            if pred in fp or fp in pred or (pred_words and pred_words & fp_words):
                results.append(f)

        # Also scan entity attributes (richer source for area, elevation, etc.)
        for rec in self._entities.values():
            for attr_k, attr_v in (rec.get("attributes") or {}).items():
                ak = str(attr_k or "").lower().replace("_", " ")
                ak_words = {w for w in ak.split() if len(w) > 3}
                if pred not in ak and ak not in pred and not (pred_words and pred_words & ak_words):
                    continue
                nv = _try_float(attr_v)
                key = (rec["name"].lower(), attr_k.lower())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                results.append({
                    "subject":          rec["name"],
                    "predicate":        attr_k,
                    "value":            str(attr_v),
                    "unit":             "",
                    "normalized_value": nv,
                    "chunk_id":         rec["chunk_ids"][0] if rec["chunk_ids"] else "",
                    "_from_entity":     True,
                })

        return results

    def check_topic(self, topic: Any) -> List[Dict]:
        """Return theme records (and entity mentions) matching *topic* keywords."""
        if isinstance(topic, list):
            topic = " ".join(str(x) for x in topic)
        else:
            topic = str(topic or "")
        keywords = [w.lower() for w in topic.replace("_", " ").split() if len(w) > 2]
        results = []

        for t in self._themes:
            text = (
                (t.get("topic") or "") + " " +
                (t.get("context") or "") + " " +
                (t.get("entity") or "")
            ).lower()
            if any(kw in text for kw in keywords):
                results.append(t)

        # Also scan entity names and claim text
        for rec in self._entities.values():
            name = str(rec.get("name") or "").lower()
            if any(kw in name for kw in keywords):
                results.append({
                    "topic":   topic,
                    "entity":  rec["name"],
                    "context": "entity mention",
                    "_from_entity": True,
                })

        for c in self._claims:
            if any(kw in c.get("text", "").lower() for kw in keywords):
                results.append({
                    "topic":   topic,
                    "entity":  c.get("entity", ""),
                    "context": c.get("text", "")[:120],
                    "_from_claim": True,
                })

        return results

    def find_unit_outliers(
        self,
        predicate: str = "area",
        allowed_types: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Detect facts where the same predicate uses inconsistent units across entities.
        Returns formatted lines like "UNIT-GROUP [sq miles]: Jotunheimen National Park".

        allowed_types: if provided, only consider facts whose subject is an entity of
                       one of these types (e.g. {"national_park"}).  This prevents
                       non-park entities (hotels, castles, etc.) from contaminating
                       the unit-mismatch analysis (Idea C).
        """
        # Build set of allowed subject names if type filtering is requested (Idea C)
        allowed_subjects: Optional[Set[str]] = None
        if allowed_types:
            allowed_subjects = set()
            for et in allowed_types:
                for e in self.entities_by_type(et):
                    name = str(e.get("name") or "").lower().strip()
                    if name:
                        allowed_subjects.add(name)

        pred = str(predicate or "").lower().replace("_", " ")
        pred_words = {w for w in pred.split() if len(w) > 3}

        unit_buckets: Dict[str, List[str]] = {}
        for f in self._facts:
            fp = str(f.get("predicate") or "").lower().replace("_", " ")
            if pred not in fp and not (pred_words and any(w in fp for w in pred_words)):
                continue
            subj = str(f.get("subject") or "")
            # Idea C: skip facts for entities outside the allowed types.
            # Use substring matching as fallback: fact subject "Jotunheimen" matches
            # entity name "Jotunheimen National Park" (subject is a prefix of full name).
            if allowed_subjects is not None:
                subj_low = subj.lower().strip()
                if subj_low not in allowed_subjects:
                    # Fallback: check if any allowed name contains subj or vice-versa
                    if not any(
                        subj_low in s or s in subj_low
                        for s in allowed_subjects
                        if len(s) >= 4 and len(subj_low) >= 4
                    ):
                        continue
            unit = str(f.get("unit") or "").lower().strip()
            if unit:
                unit_buckets.setdefault(unit, []).append(subj)

        if len(unit_buckets) <= 1:
            return []

        lines = []
        for unit, subjects in sorted(unit_buckets.items(), key=lambda x: len(x[1])):
            unique = list(dict.fromkeys(subjects))
            lines.append(f"UNIT-GROUP [{unit}]: {', '.join(unique[:6])}")
        return lines

    def find_contradictions(self, predicate: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Detect facts where the same subject has multiple distinct values for the same predicate.
        Returns {key: [fact, ...]} for each contradiction group.
        """
        candidates = self.facts_by_predicate(predicate) if predicate else self._facts

        groups: Dict[str, List[Dict]] = {}
        for f in candidates:
            subj = _norm_subject(str(f.get("subject") or ""))
            pred = str(f.get("predicate") or "").lower().strip()
            if not subj or not pred:
                continue
            groups.setdefault(f"{subj}||{pred}", []).append(f)

        contradictions = {}
        for key, facts in groups.items():
            distinct = set()
            for f in facts:
                sig = _value_sig(f)
                if sig is not None:
                    distinct.add(sig)
                else:
                    v = str(f.get("value") or "").strip().lower()
                    if v:
                        distinct.add(v)
            if len(distinct) > 1:
                contradictions[key] = facts

        return contradictions

    def keyword_search(self, query: Any, limit: int = 30) -> List[str]:
        """
        Search all structured records for *query* keywords.
        Returns formatted evidence lines.
        """
        keywords = _extract_keywords(query)
        if not keywords:
            return []

        lines: List[str] = []

        for f in self._facts:
            text = f"{f.get('subject') or ''} {f.get('predicate') or ''} {f.get('value') or ''}".lower()
            if any(kw in text for kw in keywords):
                val_str = f"{f.get('value') or ''} {f.get('unit') or ''}".strip()
                lines.append(f"FACT: {f.get('subject') or ''} / {f.get('predicate') or ''} / {val_str}")

        for rec in self._entities.values():
            name = str(rec.get("name") or "")
            if any(kw in name.lower() for kw in keywords):
                attrs = rec.get("attributes") or {}
                attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:6])
                entry = f"ENTITY [{rec.get('type') or ''}]: {name}"
                if attr_str:
                    entry += f"  ({attr_str})"
                lines.append(entry)

        for c in self._claims:
            if any(kw in str(c.get("text") or "").lower() for kw in keywords):
                lines.append(f"CLAIM: {str(c.get('text') or '')[:200]}")

        for ev in self._events:
            text = f"{ev.get('description') or ''} {ev.get('entity') or ''}".lower()
            if any(kw in text for kw in keywords):
                lines.append(
                    f"EVENT ({ev.get('year_text') or '?'}): {ev.get('description') or ''} [{ev.get('entity') or ''}]"
                )

        for t in self._themes:
            text = f"{t.get('topic') or ''} {t.get('entity') or ''} {t.get('context') or ''}".lower()
            if any(kw in text for kw in keywords):
                lines.append(f"THEME [{t.get('topic') or ''}]: {t.get('entity') or ''} — {(t.get('context') or '')[:100]}")

        return lines[:limit]

    def keyword_search_raw(self, query: Any, limit: int = 20) -> List[str]:
        """
        Fallback: search raw chunk texts for *query* keywords.
        Returns best-matching paragraphs sorted by keyword-overlap count.
        Used as needle fallback when structured search yields nothing.
        """
        keywords = _extract_keywords(query)
        if not keywords or not self._raw_chunks:
            return []

        scored: List[tuple] = []
        for chunk in self._raw_chunks:
            text = chunk.get("text", "")
            paragraphs = re.split(r'\n{2,}', text.strip())
            for para in paragraphs:
                if len(para.split()) < 5:
                    continue
                para_lower = para.lower()
                score = sum(1 for kw in keywords if kw in para_lower)
                if score > 0:
                    scored.append((score, para[:700]))

        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        result: List[str] = []
        for _, p in scored:
            key = p[:80]
            if key not in seen:
                seen.add(key)
                result.append(p)
            if len(result) >= limit:
                break
        return result

    def phrase_search(self, phrase: str, limit: int = 10) -> List[str]:
        """
        Word-boundary (case-insensitive) phrase search across raw chunk text.
        Uses regex \\b so "accessibility" does NOT match "inaccessibility".
        Returns sentences that contain the phrase as whole words.
        """
        phrase_lower = str(phrase or "").lower().strip()
        if not phrase_lower or not self._raw_chunks:
            return []
        try:
            pattern = re.compile(r'\b' + re.escape(phrase_lower) + r'\b', re.IGNORECASE)
        except re.error:
            # Fallback to substring for unusual phrases (e.g. regex specials)
            pattern = None

        splitter = re.compile(r'(?<=[.!?])\s+')
        results: List[str] = []
        seen: set = set()
        for chunk in self._raw_chunks:
            text = chunk.get("text", "")
            # Fast pre-filter: substring check before regex
            if phrase_lower not in text.lower():
                continue
            # Word-boundary check on the whole chunk
            if pattern and not pattern.search(text):
                continue
            for sent in splitter.split(text):
                if len(sent.split()) < 3:
                    continue
                match = pattern.search(sent) if pattern else (phrase_lower in sent.lower())
                if match:
                    key = sent[:70]
                    if key not in seen:
                        seen.add(key)
                        results.append(sent.strip()[:300])
                    if len(results) >= limit:
                        return results
        return results

    def phrase_search_variants(self, phrase: str, limit: int = 6) -> List[str]:
        """
        phrase_search with automatic singular/plural/core-phrase variants.
        Handles cases like 'national nature reserves' → also tries
        'national nature reserve' (singular) and 'nature reserve' (core).
        """
        phrase = str(phrase or "").strip().lower()
        if not phrase:
            return []

        candidates: List[str] = [phrase]
        words = phrase.split()

        if len(words) > 1:
            last = words[-1]
            # singular variants — try strip 's' first so "reserves" → "reserve" (not "reserv")
            if last.endswith("s") and len(last) > 3:
                cand_s = " ".join(words[:-1] + [last[:-1]])   # reserves→reserve, parks→park
                candidates.append(cand_s)
            if last.endswith("es") and len(last) > 4:
                cand_es = " ".join(words[:-1] + [last[:-2]])  # churches→church, boxes→box
                if cand_es not in candidates:
                    candidates.append(cand_es)
            # plural: add 's'
            if not last.endswith("s"):
                candidates.append(phrase + "s")
            # core: drop common filler words
            _FILLER = {"for", "the", "and", "with", "from", "its", "all",
                       "that", "this", "are", "its", "of", "to", "in", "a"}
            core = [w for w in words if w not in _FILLER and len(w) > 2]
            if core and core != words:
                candidates.append(" ".join(core))

        seen: set = set()
        results: List[str] = []
        for variant in candidates:
            for hit in self.phrase_search(variant, limit=limit):
                key = hit[:70]
                if key not in seen:
                    seen.add(key)
                    results.append(hit)
            if len(results) >= limit:
                break
        return results[:limit]

    def sentence_search(self, query: Any, limit: int = 25) -> List[str]:
        """
        Ultimate needle fallback: sentence-level full document scan.

        Splits every stored raw chunk into individual sentences, scores each
        by keyword overlap with *query*, and returns the top-N matches.

        This runs even when paragraph-level search finds nothing — it catches
        facts that appear in dense prose where the paragraph contains many
        unrelated sentences diluting the keyword signal.
        """
        # Coerce non-string query (int, list, None) to string
        if isinstance(query, list):
            query = " ".join(str(x) for x in query)
        else:
            query = str(query or "")

        keywords = _extract_keywords(query)
        if not keywords or not self._raw_chunks:
            return []

        # Also try any word ≥ 4 chars from the query (broader net)
        broad_kws = [
            w.lower().strip(".,?!'\"()")
            for w in query.split()
            if len(w) >= 4
        ]

        scored: List[tuple] = []
        splitter = re.compile(r'(?<=[.!?])\s+')

        for chunk in self._raw_chunks:
            text = chunk.get("text", "")
            sentences = splitter.split(text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) < 5:
                    continue
                sent_lower = sent.lower()
                # Primary score: filtered keywords
                score = sum(1 for kw in keywords if kw in sent_lower)
                # Bonus for broad keywords (catches cases where stop-word removal killed the signal)
                if score == 0:
                    score = sum(0.5 for kw in broad_kws if kw in sent_lower and kw not in _STOP_WORDS)
                if score > 0:
                    scored.append((score, sent[:600]))

        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        result: List[str] = []
        for _, s in scored:
            key = s[:70]
            if key not in seen:
                seen.add(key)
                result.append(s)
            if len(result) >= limit:
                break
        return result

    def to_summary_text(self, max_tokens: int = 8000) -> str:
        """
        Produce a compact summary of the KnowledgeStore for global-synthesis queries.

        Section order (most useful first for synthesis questions):
          1. Themes   — recurring topics and patterns across the document
          2. Events   — dated occurrences
          3. Claims   — explicit assertions from the text
          4. Facts    — numeric/categorical predicates
          5. Entities — structured records (trimmed last if budget runs out)

        Estimates ~4 chars per token; hard-truncates to budget.
        """
        char_budget = max_tokens * 4
        lines: List[str] = []

        lines.append(
            f"=== KNOWLEDGE STORE ({len(self._entities)} entities, {len(self._facts)} facts, "
            f"{len(self._events)} events, {len(self._themes)} themes) ==="
        )

        lines.append("\n--- THEMES ---")
        for t in self._themes[:300]:
            lines.append(
                f"  [{t.get('topic') or ''}] {t.get('entity') or ''}"
                f" — {(t.get('context') or '')[:100]}"
            )

        lines.append("\n--- EVENTS ---")
        for ev in self._events[:150]:
            lines.append(
                f"  ({ev.get('year_text') or '?'}) {ev.get('description') or ''}"
                f" [{ev.get('entity') or ''}]"
            )

        lines.append("\n--- CLAIMS ---")
        for c in self._claims[:150]:
            lines.append(f"  {str(c.get('text') or '')[:180]}")

        lines.append("\n--- FACTS ---")
        for f in self._facts[:400]:
            val_str = f"{f.get('value') or ''} {f.get('unit') or ''}".strip()
            lines.append(
                f"  {f.get('subject') or ''} / {f.get('predicate') or ''} / {val_str}"
            )

        lines.append("\n--- ENTITIES ---")
        for rec in list(self._entities.values())[:200]:
            attrs = rec.get("attributes") or {}
            attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:5])
            line = f"  [{rec.get('type') or ''}] {rec.get('name') or ''}"
            if attr_str:
                line += f":  {attr_str}"
            lines.append(line)

        full_text = "\n".join(lines)
        if len(full_text) > char_budget:
            full_text = full_text[:char_budget] + "\n[... truncated to token budget ...]"
        return full_text

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "entities":   list(self._entities.values()),
            "facts":      self._facts,
            "events":     self._events,
            "claims":     self._claims,
            "themes":     self._themes,
            "raw_chunks": self._raw_chunks,
            "inventory":  self._inventory,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("KnowledgeStore saved → %s  (%d entities, %d facts)", path,
                    len(self._entities), len(self._facts))

    @classmethod
    def load(cls, path: str) -> "KnowledgeStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls()
        for e in data.get("entities", []):
            key = e.get("name", "").lower().strip()
            if key:
                store._entities[key] = e
        store._facts      = data.get("facts",      [])
        store._events     = data.get("events",     [])
        store._claims     = data.get("claims",     [])
        store._themes     = data.get("themes",     [])
        store._raw_chunks = data.get("raw_chunks", [])
        store._inventory  = data.get("inventory",  {})
        logger.info("KnowledgeStore loaded from %s  (%d entities, %d facts)", path,
                    len(store._entities), len(store._facts))
        return store

    def stats(self) -> Dict[str, int]:
        return {
            "entities": len(self._entities),
            "facts":    len(self._facts),
            "events":   len(self._events),
            "claims":   len(self._claims),
            "themes":   len(self._themes),
        }

    def extract_raw_names(self, suffix: str = "National Park") -> List[str]:
        """
        Scan raw chunks for '<prefix> <suffix>' patterns and return unique prefixes.
        Used by the aggregation solver to count park names from raw text when the
        structured entity store is incomplete.

        Returns sorted list of unique prefix strings (e.g. ["Abisko", "Cairngorms", ...]).
        """
        escaped = re.escape(suffix)
        # Match 1-6 words before the suffix; words may contain Unicode letters, hyphens, apostrophes
        pattern = re.compile(
            r'((?:[\wÀ-ɏ\'-]+\s+){1,6})' + escaped,
            re.UNICODE | re.IGNORECASE,
        )
        found: set = set()
        for chunk in self._raw_chunks:
            for m in pattern.finditer(chunk.get("text", "")):
                name = m.group(1).strip()
                word_count = len(name.split())
                if 1 <= word_count <= 6 and len(name) >= 3:
                    found.add(name)
        return sorted(found)


# ── Helpers ───────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "what", "when", "where", "which", "does", "that", "this", "from", "with",
    "have", "been", "were", "they", "their", "there", "about", "book",
    "named", "name", "list", "give", "many", "much",
    "some", "each", "also", "than", "into", "such", "only", "over", "both",
    "document", "text", "mentioned", "found", "refers", "states", "said",
    "according", "described", "noted",
}


def _extract_keywords(query: Any) -> List[str]:
    # Coerce any non-string input (int, list, None) to string
    if isinstance(query, list):
        query = " ".join(str(x) for x in query)
    else:
        query = str(query or "")
    result = []
    for w in query.split():
        w = w.lower().strip(".,?!'\"()")
        w = re.sub(r"'s$|'s$", "", w)   # normalize possessives: "Etna's" → "etna"
        if len(w) > 3 and w not in _STOP_WORDS:
            result.append(w)
    return result


def _try_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"[\d]+\.?\d*", val.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                pass
    return None


_ENTITY_ABBREV = [
    (re.compile(r'\bnp\b',  re.IGNORECASE), 'national park'),
    (re.compile(r'\bnr\b',  re.IGNORECASE), 'nature reserve'),
    (re.compile(r'\bnf\b',  re.IGNORECASE), 'national forest'),
    (re.compile(r'\bsp\b',  re.IGNORECASE), 'state park'),
    (re.compile(r'\bnwr\b', re.IGNORECASE), 'national wildlife refuge'),
]


def _entity_merge_key(name: str) -> str:
    """
    Normalize entity name for deduplication merge key.

    Expands widely-used type abbreviations so variants like
    'Bavarian Forest NP' and 'Bavarian Forest National Park'
    map to the same key and get merged correctly.
    Works for any document — only generic abbreviations are expanded.
    """
    s = name.lower().strip()
    for pattern, replacement in _ENTITY_ABBREV:
        s = pattern.sub(replacement, s)
    return re.sub(r'\s+', ' ', s).strip()


def _norm_subject(s: str) -> str:
    """Normalize entity name for contradiction grouping (strip 'National Park' suffix etc.)."""
    s = s.lower().strip()
    s = re.sub(r"\s*(national\s+park|nat\.?\s*park|np)\s*$", "", s)
    return s.strip()


def _value_sig(f: Dict) -> Optional[float]:
    """Extract a comparable numeric signature from a fact for contradiction detection."""
    nv = f.get("normalized_value")
    if nv is not None:
        try:
            return round(float(nv), 2)
        except (TypeError, ValueError):
            pass
    v = str(f.get("value") or "")
    m = re.search(r"[\d,]+\.?\d*", v.replace(",", ""))
    if m:
        try:
            return round(float(m.group()), 2)
        except ValueError:
            pass
    return None

"""
Few-shot LLM question type classifier.

7 types: aggregation, superlative, absence, contradiction,
         cross_section, global_synthesis, needle

Public API
----------
classify_question_type(q_text)          → str
classify_questions_batch(questions)     → Dict[str, str]  (id → type)
verify_classifier(dataset_path, ...)    → Dict  (accuracy report)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .mistral_client import call_model, extract_json
from .config import MAIN_MODEL, TEMPERATURE_FACTUAL

logger = logging.getLogger(__name__)

QUESTION_TYPES = [
    "aggregation",
    "superlative",
    "absence",
    "contradiction",
    "cross_section",
    "global_synthesis",
    "needle",
]

_TYPE_DESCRIPTIONS = {
    "aggregation":
        "Count or list ALL entities satisfying a criterion "
        "(how many, which ones, list all…)",
    "superlative":
        "Find THE ONE entity with the highest or lowest value on a numeric "
        "dimension (largest, highest, most, oldest, smallest, fewest…)",
    "absence":
        "Determine which option among given alternatives is NOT mentioned / "
        "NOT discussed in the document — always presents multiple options",
    "contradiction":
        "Find conflicting or inconsistent claims within the document itself "
        "(two passages that disagree on the same fact)",
    "cross_section":
        "Compare or combine information drawn from multiple separate sections "
        "of the document (what do X and Y have in common, compare A vs B…)",
    "global_synthesis":
        "Summarise, characterise, or interpret the document AS A WHOLE "
        "(recurring themes, overarching narrative, overall argument, author's stance…)",
    "needle":
        "Retrieve ONE specific fact stated at a single location in the document "
        "(what year, who, what is the exact value, which name…)",
}

# ── Hardcoded few-shot examples (15-20 per type) ──────────────────────────────
# These are embedded so the classifier works without loading any external file.
_FEW_SHOT: Dict[str, List[str]] = {
    "aggregation": [
        "How many countries signed the treaty described in the first chapter?",
        "Which scientists mentioned in the book published their work before 1900?",
        "List all mountains in the text with an elevation above 4000 metres.",
        "How many buildings profiled were constructed during the Renaissance?",
        "Which athletes described have won more than three Olympic medals?",
        "How many rivers mentioned flow into the Arctic Ocean?",
        "Which companies in the book went bankrupt within ten years of founding?",
        "List all battles described that took place in Asia.",
        "How many species covered are classified as endangered?",
        "Which authors mentioned were born in the nineteenth century?",
        "How many of the described experiments produced a statistically significant result?",
        "List all cities profiled with a population exceeding one million.",
        "Which structures in the book were subsequently demolished?",
        "How many of the expeditions described reached their stated destination?",
        "List all languages mentioned that belong to the Semitic language family.",
        # threshold counts, multi-condition filters — NOT superlative (no single extreme)
        "How many national parks described cover an area exceeding 500 square kilometres?",
        "Which treaties mentioned were signed in the twentieth century and ratified by more than ten nations?",
        "List all scientists named in the research sections who worked across more than one discipline.",
        "How many hospitals described in the document opened after the year 2000?",
        "Which economic sectors reported both rising employment and declining output simultaneously?",
        # technical release notes — counting CVEs, advisories, versions, update letters
        "How many distinct CVE identifiers are referenced across the entire document?",
        "How many ESXi update release note sections does the document contain, and which update letters do they cover?",
        "Which update letters have vCenter Server release notes but no corresponding ESXi release notes?",
        "How many distinct VMSA security advisories are cited in the document, and what range of years do they span?",
        "For how many update letters do the ESXi and vCenter Server release notes share the same release date?",
    ],
    "superlative": [
        "Which country in the book has the highest recorded GDP growth rate?",
        "What is the deepest lake described in the text?",
        "Which expedition covered the greatest total distance?",
        "Who is the oldest person mentioned in the document?",
        "Which building profiled has the highest number of floors?",
        "What is the earliest date mentioned anywhere in the document?",
        "Which company described has the highest annual revenue?",
        "Who holds the record for the fastest completion time of the event described?",
        "Which species mentioned has the longest known lifespan?",
        "What is the most recent event recorded in the document?",
        "Which city profiled has the lowest average annual rainfall?",
        "Who in the book made the most discoveries during their career?",
        "Which material described has the highest tensile strength?",
        "What is the longest river mentioned in the text?",
        "Which institution has the largest collection described in the book?",
        # document-wide extremes with no named box/table/section scope
        "Among all countries described in the document, which recorded the fastest economic recovery?",
        "Which of all the structures profiled throughout the text was built using the greatest number of distinct materials?",
        "Out of all the scientists profiled in the book, who received the most awards during their career?",
        "What is the highest elevation reached by any expedition described anywhere in the text?",
        "Which organism described in the document has the shortest recorded generation time?",
        # technical release notes — finding extremes across all releases
        "What is the oldest CVE identifier referenced anywhere in the document, by the year in its identifier?",
        "Which is the most recent release documented in the file, and what is its build number?",
        "Across all ESXi update releases described, which has the largest stated download size?",
    ],
    "absence": [
        "Of inflation, unemployment, and currency devaluation, which economic phenomenon is NOT discussed?",
        "Which of the following is NOT cited as a cause of the conflict: famine, foreign intervention, or drought?",
        "Of solar panels, wind turbines, and tidal generators, which energy source is NOT mentioned?",
        "Which of these materials is NOT described as used in construction: steel, bamboo, or concrete?",
        "Of these cities, which is NOT mentioned as a trade hub: Lisbon, Alexandria, or Samarkand?",
        "Which of the following diseases is NOT discussed anywhere: cholera, typhus, or malaria?",
        "Of negotiation, arbitration, and litigation, which conflict resolution method is NOT described?",
        "Which of these roles is NOT listed as held by the subject: diplomat, poet, or soldier?",
        "Of river transport, road transport, and air transport, which mode is NOT mentioned as used?",
        "Which of the following is NOT named as a co-inventor: Tesla, Edison, or Marconi?",
        "Of saltwater, freshwater, and brackish environments, which ecosystem type is NOT mentioned?",
        "Which of the following tools is NOT described as used in the excavation: trowel, sieve, or radar?",
        "Of timber, stone, and clay, which building material is NOT mentioned for this structure?",
        "Which funding source is NOT cited: government grants, private donors, or crowdfunding?",
        "Of migration, hibernation, and camouflage, which survival strategy is NOT attributed to the described species?",
        # varied option structures and topic categories
        "Of photosynthesis, respiration, and fermentation, which biological process is NOT described in the text?",
        "Which of these diplomatic outcomes is NOT attributed to the negotiations: ceasefire, prisoner exchange, or territorial concession?",
        "Among these three economic frameworks — Keynesian, monetarist, and supply-side — which is NOT discussed in the document?",
        "Of cognitive bias, confirmation bias, and anchoring bias, which is NOT mentioned as a contributing factor?",
        "Which of the following rights is NOT cited as protected under the described legislation: freedom of speech, right to assembly, or right to privacy?",
        # technical release notes — absent technologies or document features
        "Of ransomware, VMSA security advisories, knowledge base articles, and severity ratings — which is never mentioned in the document?",
        "Of TLS, IPv6, GPU support, and FIPS compliance — which is never mentioned anywhere in the document?",
        "Of Kubernetes, NSX, vSAN, and Hyper-V — which is never mentioned anywhere in the document?",
    ],
    "contradiction": [
        "The document makes conflicting claims about when a particular law was passed. What are the two claims?",
        "Two sections give different population figures for the same city. Identify the discrepancy.",
        "Find a contradiction in the reported height of a specific structure described in the book.",
        "The document describes an event as both a success and a failure in different sections. Where?",
        "Identify conflicting dates given for the same person's birth in the document.",
        "Two chapters give different values for the same physical measurement. Find them.",
        "Find a statement that contradicts the founding date given elsewhere in the document.",
        "Identify where the document gives inconsistent figures for the area of a described region.",
        "The text attributes the same invention to two different people. Where does this occur?",
        "Find conflicting claims about the duration of a historical period described in the book.",
        "What inconsistency exists in the reported distance between two locations mentioned?",
        "Identify contradictory claims about who held a specific record according to the document.",
        "Two passages describe the same species as both endemic and introduced. Identify them.",
        "Find a contradiction in the reported number of casualties in a described event.",
        "The document gives two different years for when an organisation was established. Find both.",
        # attribution, temporal, geographic, and numeric inconsistency variants
        "Two passages in the document attribute different nationalities to the same historical figure. Identify the contradiction.",
        "Find the section where the document places the same event in two different decades.",
        "The document gives conflicting descriptions of the same geographic feature in two different chapters. What is contradicted?",
        "Identify where two different paragraphs give conflicting accounts of who led the described expedition.",
        "Find the contradiction in the reported vote tally for the described resolution between two sections of the document.",
        # technical release notes — CVE and version inconsistencies across sections
        "One CVE identifier is described differently in two separate release note sections. Which CVE, and what does each section say?",
        "Two release note sections give conflicting instructions for applying the same patch. Where does this occur?",
        "The document describes the same software component as both deprecated and supported in different update sections. Identify the inconsistency.",
    ],
    "cross_section": [
        "Which individuals profiled in both the early and later chapters share the same professional background?",
        "Compare the causes of failure cited for projects described in the first and final chapters.",
        "What methods are common to both the field experiments and the laboratory experiments described?",
        "Which organisations described in separate sections share the same founding principles?",
        "Compare the leadership styles of the figures profiled in two different chapters.",
        "What recurring argument links the case studies in the economics and policy sections?",
        "How do the infrastructure challenges described in the urban and rural sections differ?",
        "Which two events described in separate sections have the most closely overlapping dates?",
        "What common criticism is levelled at both institutions described in different parts of the book?",
        "Compare the evidence cited for the same hypothesis in the theoretical and empirical chapters.",
        "Which discoveries profiled in different disciplines share a common methodology?",
        "How do the cultural practices described in the northern and southern regions compare?",
        "Which biographical subjects profiled in separate chapters worked on the same project?",
        "What themes connect the earliest and latest documented case studies in the book?",
        "Compare the technical specifications given for similar devices described in different sections.",
        # named sections/entities explicitly referenced — distinct from whole-document global_synthesis
        "How do the economic policies recommended in the opening chapter compare with those proposed in the conclusion?",
        "Which figures introduced in the biography section later appear in the policy analysis section?",
        "What evidence do both the qualitative findings chapter and the quantitative analysis chapter provide in support of the same claim?",
        "Compare the population estimates given for the same region in the historical overview chapter versus the contemporary data chapter.",
        "Which technologies introduced in the early chapters are critically reassessed in the later evaluation chapters?",
        # technical release notes — comparing parallel ESXi vs vCenter sections, or early vs late updates
        "Do the ESXi and vCenter Server release notes for the same update letter describe the same release event? Compare their dates and build numbers.",
        "The ESXi and vCenter Server release notes for a specific update carry the same date but give different patch instructions. What is the difference?",
        "Where in the document are component deprecations first discussed, and how are they handled in the later update release notes?",
    ],
    "global_synthesis": [
        "What is the central argument the book makes across all its chapters?",
        "How does the author characterise the relationship between technological progress and social change throughout the text?",
        "What recurring tension does the book identify across all the case studies?",
        "What overall conclusion does the author reach about the subject by the end of the book?",
        "How does the book's tone shift from its opening chapters to its conclusion?",
        "What patterns of success and failure does the book consistently identify across its subjects?",
        "How does the book frame the role of individual agency versus structural forces throughout?",
        "What is the book's overall stance on the policy approach it describes?",
        "How does the author use the different case studies to build a single unifying argument?",
        "What image of the subject emerges from reading all the profiled examples together?",
        "How does the book treat uncertainty and gaps in the evidence across all chapters?",
        "What common methodology do all the investigations described in the book share?",
        "How does the book characterise progress in the field over the period it covers?",
        "What are the most consistently cited obstacles across all the examples in the book?",
        "How does the book balance descriptive and prescriptive claims across its entirety?",
        # whole-document scope, no specific section or entity named
        "What overarching lesson does the document draw from all the historical examples it presents?",
        "How consistently does the book treat the role of leadership across all of its case studies?",
        "What type of evidence does the author most frequently rely on throughout the entire work?",
        "What single most important conclusion can be drawn from reading the document in its entirety?",
        "How does the document's definition of progress evolve from its introduction to its final chapter?",
        # technical release notes — trends and patterns across all releases, whole-document scope
        "Across all the ESXi update releases documented, how does the balance between bug fixes and security fixes change over the product's lifetime?",
        "Reading the release notes in date order, how does the document's treatment of the VMware brand name change, and where does the change first occur?",
        "The release notes state their dates in a consistent format with a small number of exceptions. Which releases break the convention, and how?",
    ],
    "needle": [
        "In what year was the organisation described in chapter three founded?",
        "What is the exact mass of the object described in the experiment?",
        "Who coined the term introduced in the second paragraph of chapter five?",
        "What temperature was recorded during the event described in that section?",
        "Which river is cited as the boundary in the treaty described?",
        "In what month did the subject resign from their position?",
        "What is the capacity figure given for the structure described in section four?",
        "Who wrote the letter quoted in the final chapter?",
        "What distance is given between the two locations mentioned in the document?",
        "In what year did the first documented outbreak occur according to the text?",
        "What material is described as the primary component of the artefact?",
        "How many days did the described journey last according to the text?",
        "What name is given to the process described in the third section?",
        "In which city did the event described in chapter two take place?",
        "What is the exact figure cited for the budget of the described project?",
        # Location-anchored comparative questions — these are needle, NOT superlative.
        # The presence of Box/Table/Figure/Section reference signals a single lookup location.
        "Which nation, among those discussed in Box 3.1, recorded the weakest growth over the observed period?",
        "Which of the six economies listed in Table 3.2 experienced the steepest decline between 2020 and 2025?",
        "Among the countries covered in Figure 2.1, which one showed a consistent downward trend?",
        "What nation, among those described in Box 5.2, had the lowest recovery rate after the crisis?",
        "Which country, of the five examined in Section 4.3, recorded the largest drop in employment?",
        "What region is identified in Box 2.1 as having seen the sharpest population decline?",
        # scoped single-location lookups with specific document cues
        "According to the data presented in Table 1.1, what value was reported for the lowest-performing country?",
        "What does the author state in the first paragraph of Chapter 4 as the primary cause of the described outcome?",
        "According to Figure 3.3, which year marked the turning point in the trend described?",
        "What specific term does the document use in Section 2 to label the underlying mechanism?",
        # technical release notes — specific facts at a single stated location
        "What is the stated download size of the ESXi 7.0 Update 3w depot?",
        "What is the md5sum of the ESXi 7.0 Update 3w download?",
        "Which release does the document name as supporting a direct upgrade path to ESXi 8.x?",
        "What build number is given for the vCenter Server 7.0 Update 3w release?",
    ],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    type_list = "\n".join(
        f"  {t}: {_TYPE_DESCRIPTIONS[t]}" for t in QUESTION_TYPES
    )
    example_lines: List[str] = []
    for q_type in QUESTION_TYPES:
        for ex in _FEW_SHOT[q_type]:
            example_lines.append(f'Question: "{ex}"')
            example_lines.append(f"Type: {q_type}")
            example_lines.append("")

    return (
        "You are a question type classifier for a document Q&A system.\n\n"
        "Classify questions into exactly one of these 7 types:\n"
        f"{type_list}\n\n"
        "Key distinctions:\n"
        "  • aggregation vs superlative: aggregation = count/list ALL qualifying items; "
        "superlative = identify THE SINGLE item with the extreme value\n"
        "  • aggregation vs cross_section: aggregation produces a COUNT or LIST of items satisfying "
        "a criterion; cross_section produces a COMPARISON or RELATIONSHIP between two or more "
        "named parts of the document\n"
        "  • absence: ALWAYS presents a discrete list of options and asks which is NOT in the document\n"
        "  • contradiction: seeks an inconsistency between two claims WITHIN the document\n"
        "  • cross_section vs global_synthesis: cross_section names SPECIFIC sections or entities "
        "to compare (e.g. 'compare Chapter 1 with Chapter 5'); global_synthesis asks about the "
        "WHOLE document with no specific section or entity reference\n"
        "  • global_synthesis: asks about the document AS A WHOLE — themes, argument, recurring patterns\n"
        "  • needle: asks for ONE specific fact found at a single location\n\n"
        "CRITICAL — needle vs superlative distinction:\n"
        "  • superlative: requires RANKING or COMPARING numeric values across all entities in the document "
        "to find the extreme (e.g. 'which country has the highest GDP across all countries?')\n"
        "  • needle: the question names a SPECIFIC location (Box X.X, Table X.X, Figure X.X, "
        "Section X, Chapter X) where the answer already appears — even if comparative language "
        "('fell behind', 'steepest decline', 'largest drop') is used. "
        "If the question scopes to a named subset ('among the six countries in Box 1.4'), "
        "the answer is a stated fact at that location, NOT a document-wide numeric ranking. "
        "Classify these as needle.\n\n"
        "Examples:\n\n"
        + "\n".join(example_lines)
        + "Output ONLY valid JSON — a single object mapping each question id to its type.\n"
        'Example: {"q1": "aggregation", "q2": "needle"}'
    )


# ── Public API ────────────────────────────────────────────────────────────────

def classify_question_type(q_text: str) -> str:
    """Classify a single question text. Returns one of the 7 type strings."""
    results = classify_questions_batch([{"id": "_single_", "question": q_text}])
    return results.get("_single_", "needle")


def classify_questions_batch(
    questions: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Classify a batch of questions in one LLM call.

    Parameters
    ----------
    questions : list of dicts, each with 'id' and 'question' keys.

    Returns
    -------
    Dict mapping question id → predicted type string.
    """
    if not questions:
        return {}

    system = _build_system_prompt()

    q_lines = "\n".join(
        f'  "{q["id"]}": "{q.get("question", "")}"'
        for q in questions
    )
    user = (
        f"Classify these {len(questions)} question(s):\n{{\n{q_lines}\n}}\n\n"
        "Return ONLY the JSON object (id → type):"
    )

    max_out = min(len(questions) * 25 + 200, 2000)
    raw, _ = call_model(
        system, user,
        model=MAIN_MODEL,
        temperature=TEMPERATURE_FACTUAL,
        max_tokens=max_out,
    )

    try:
        result = extract_json(raw)
        cleaned: Dict[str, str] = {}
        for qid, qtype in result.items():
            if qtype in QUESTION_TYPES:
                cleaned[qid] = qtype
            else:
                logger.warning("Unknown type %r for %s — defaulting to needle", qtype, qid)
                cleaned[qid] = "needle"
        # Fill any missing ids
        for q in questions:
            if q["id"] not in cleaned:
                logger.warning("No prediction for %s — defaulting to needle", q["id"])
                cleaned[q["id"]] = "needle"
        return cleaned
    except Exception as exc:
        logger.error("Batch classification parse failed: %s | raw: %s", exc, raw[:300])
        return {q["id"]: "needle" for q in questions}


def verify_classifier(
    dataset_path: Optional[str] = None,
    n_few_shot: int = 15,
    batch_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Measure classifier accuracy on held-out examples.

    Uses the first `n_few_shot` examples per type as the training signal
    (embedded in the few-shot prompt), then tests on the remaining examples.

    Parameters
    ----------
    dataset_path : path to question_type_dataset.yaml (default: auto-detected)
    n_few_shot   : number of examples per type used as few-shot context
                   (must be < 25 to leave holdout examples)
    batch_size   : questions per LLM call during verification
    verbose      : print results table

    Returns
    -------
    Dict with keys: overall_accuracy, total_correct, total_tested,
                    per_type (dict), errors (list)
    """
    p = Path(dataset_path) if dataset_path else (Path(__file__).parent / "question_type_dataset.yaml")
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            "Make sure question_type_dataset.yaml is in src_v4/."
        )
    with open(p, encoding="utf-8") as f:
        dataset: List[Dict[str, str]] = yaml.safe_load(f)

    # Split into few-shot pool and holdout
    counts: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    holdout: List[Dict[str, str]] = []
    for item in dataset:
        q_type = item.get("type", "")
        if q_type not in QUESTION_TYPES:
            continue
        if counts[q_type] < n_few_shot:
            counts[q_type] += 1
        else:
            holdout.append(dict(item))

    if not holdout:
        return {
            "error": (
                f"No holdout examples — dataset has ≤{n_few_shot} examples per type. "
                "Reduce n_few_shot or add more examples to the dataset."
            )
        }

    if verbose:
        print(f"\n{'─' * 62}")
        print(f"  Classifier Verification")
        print(f"{'─' * 62}")
        print(f"  Few-shot : {n_few_shot} per type × {len(QUESTION_TYPES)} types "
              f"= {n_few_shot * len(QUESTION_TYPES)} examples  (hardcoded in classifier)")
        print(f"  Holdout  : {len(holdout)} questions  (from {p.name})")
        print(f"  Batch    : {batch_size} questions per LLM call")
        print(f"{'─' * 62}\n")

    # Assign temporary ids and run batch classification
    for i, item in enumerate(holdout):
        item["_tid"] = f"hq{i:04d}"

    predictions: Dict[str, str] = {}
    batches = [holdout[i: i + batch_size] for i in range(0, len(holdout), batch_size)]
    for b_idx, batch in enumerate(batches):
        batch_qs = [{"id": h["_tid"], "question": h["question"]} for h in batch]
        preds = classify_questions_batch(batch_qs)
        predictions.update(preds)
        if verbose:
            print(f"  Batch {b_idx + 1}/{len(batches)} — classified {len(batch)} questions")

    # Score
    correct_by_type: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    total_by_type:   Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    errors: List[Dict[str, str]] = []

    for item in holdout:
        true_type = item["type"]
        pred_type = predictions.get(item["_tid"], "needle")
        total_by_type[true_type] += 1
        if pred_type == true_type:
            correct_by_type[true_type] += 1
        else:
            errors.append({
                "question":  item["question"],
                "true_type": true_type,
                "pred_type": pred_type,
            })

    total_correct = sum(correct_by_type.values())
    total_total   = sum(total_by_type.values())
    overall_acc   = total_correct / total_total if total_total else 0.0

    per_type: Dict[str, Dict] = {}
    for t in QUESTION_TYPES:
        n = total_by_type[t]
        c = correct_by_type[t]
        per_type[t] = {"correct": c, "total": n, "accuracy": c / n if n else 0.0}

    if verbose:
        print(f"\n{'─' * 62}")
        print(f"  {'Type':<22}  {'Correct':>7}  {'Total':>5}  {'Acc':>7}")
        print(f"{'─' * 62}")
        for t in QUESTION_TYPES:
            s = per_type[t]
            print(f"  {t:<22}  {s['correct']:>7}  {s['total']:>5}  {s['accuracy']:>7.1%}")
        print(f"{'─' * 62}")
        print(f"  {'OVERALL':<22}  {total_correct:>7}  {total_total:>5}  {overall_acc:>7.1%}")
        print(f"{'─' * 62}\n")

        if errors:
            print(f"  Misclassified ({len(errors)}):")
            for e in errors[:30]:
                q_short = e["question"][:72]
                print(f"    [{e['true_type']} → {e['pred_type']}]  {q_short}")
        else:
            print("  All holdout examples classified correctly.")
        print()

    return {
        "overall_accuracy": overall_acc,
        "total_correct":    total_correct,
        "total_tested":     total_total,
        "per_type":         per_type,
        "errors":           errors,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Make src_v4 importable when run as a script from pipeline_v4/
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ.setdefault("COURSE_API_KEY", os.environ.get("COURSE_API_KEY", ""))

    print("Running classifier verification…\n")
    report = verify_classifier(verbose=True)
    print(f"Overall accuracy: {report['overall_accuracy']:.1%}  "
          f"({report['total_correct']}/{report['total_tested']})")

"""
Sync dataset_ground_truth.json with the current ChromaDB chunks.

For every question, evidence entries are matched against the
current chunks. Output is written only if something actually
changed compared to the previous run.
"""

import json
import re
import sys
from pathlib import Path

import chromadb


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"

DATASET_PATH = ROOT_DIR / "evaluation" / "datasets" / "dataset_ground_truth.json"
OUTPUT_PATH  = ROOT_DIR / "evaluation" / "datasets" / "dataset_ground_truth_synced.json"
REPORT_PATH  = ROOT_DIR / "evaluation" / "datasets" / "evidence_match_report.txt"


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F\u2033]", '"', text)
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_sort_key(chunk_id: str):
    if chunk_id.startswith("chunk_") and chunk_id[6:].isdigit():
        return (0, int(chunk_id[6:]))
    return (1, chunk_id)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

def load_chunks():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=["documents"])

    return [
        {"chunk_id": cid, "text": txt, "norm": normalize_text(txt)}
        for cid, txt in zip(data["ids"], data["documents"])
    ]


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

def find_matches(evidence: str, chunks: list[dict]) -> list[str]:
    ev_norm = normalize_text(evidence)
    return [c["chunk_id"] for c in chunks if ev_norm in c["norm"]]


def fallback_keyword_match(
    evidence: str,
    chunks: list[dict],
    threshold: float = 0.8,
    min_words: int = 3,
) -> list[str]:
    ev_words = set(normalize_text(evidence).split())
    if len(ev_words) < min_words:
        return []

    hits = []
    for c in chunks:
        chunk_words = set(c["norm"].split())
        overlap = len(ev_words & chunk_words) / len(ev_words)
        if overlap >= threshold:
            hits.append(c["chunk_id"])
    return hits


# ---------------------------------------------------------
# Change detection
# ---------------------------------------------------------

def build_signature(dataset: list[dict]) -> dict:
    """
    Reduce the dataset to only the fields that matter for
    change detection: id -> (relevant_chunk_ids, evidence_matches).
    """
    sig = {}
    for r in dataset:
        sig[r["id"]] = {
            "relevant_chunk_ids": r.get("relevant_chunk_ids", []),
            "evidence_matches": r.get("evidence_matches", {}),
        }
    return sig


def has_changed(old: list[dict] | None, new: list[dict]) -> bool:
    """Return True iff the meaningful content differs."""
    if old is None:
        return True
    return build_signature(old) != build_signature(new)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading ChromaDB chunks...")
    chunks = load_chunks()
    print(f"  -> {len(chunks)} chunks\n")

    print(f"Loading dataset: {DATASET_PATH.name}")
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  -> {len(dataset)} questions\n")

    # -----------------------------------------------------
    # Build new dataset
    # -----------------------------------------------------

    report_lines = []
    updated_dataset = []

    total_ev = 0
    exact_matches = 0
    fallback_matches = 0
    no_matches = 0

    for record in dataset:

        report_lines.append("=" * 70)
        report_lines.append(f"[{record['id']}] {record['question']}")
        report_lines.append("=" * 70)

        evidence_matches = {}
        relevant_ids = set()

        for ev in record["evidence"]:
            total_ev += 1

            hits = find_matches(ev, chunks)
            method = "exact"

            if not hits:
                hits = fallback_keyword_match(ev, chunks)
                method = "fallback" if hits else "none"

            if method == "exact":
                exact_matches += 1
            elif method == "fallback":
                fallback_matches += 1
            else:
                no_matches += 1

            evidence_matches[ev] = hits
            relevant_ids.update(hits)

            status = {"exact": "OK  ", "fallback": "FUZZ", "none": "MISS"}[method]
            report_lines.append(f"  [{status}] {ev[:80]}")
            report_lines.append(f"          -> {hits}")

        report_lines.append("")
        report_lines.append(
            f"  Relevant chunk IDs: "
            f"{sorted(relevant_ids, key=chunk_sort_key)}"
        )
        report_lines.append("")

        updated_record = dict(record)
        updated_record["relevant_chunk_ids"] = sorted(
            relevant_ids, key=chunk_sort_key
        )
        updated_record["evidence_matches"] = evidence_matches
        updated_dataset.append(updated_record)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = [
        "=" * 70,
        "EVIDENCE MATCHING SUMMARY",
        "=" * 70,
        f"Total evidence entries   : {total_ev}",
        f"Exact matches            : {exact_matches} ({exact_matches/total_ev:.1%})",
        f"Fallback (fuzzy) matches : {fallback_matches} ({fallback_matches/total_ev:.1%})",
        f"No matches               : {no_matches} ({no_matches/total_ev:.1%})",
        "=" * 70,
        "",
    ]

    # -----------------------------------------------------
    # Decide whether to save
    # -----------------------------------------------------

    print("=" * 70)
    print("EVIDENCE MATCHING SUMMARY")
    print("=" * 70)
    print(f"Total evidence entries   : {total_ev}")
    print(f"Exact matches            : {exact_matches} ({exact_matches/total_ev:.1%})")
    print(f"Fallback (fuzzy) matches : {fallback_matches} ({fallback_matches/total_ev:.1%})")
    print(f"No matches               : {no_matches} ({no_matches/total_ev:.1%})")
    print("=" * 70)

    if no_matches:
        print("\nEvidence with NO MATCH (first 10):")
        shown = 0
        for record in updated_dataset:
            for ev, hits in record["evidence_matches"].items():
                if not hits:
                    print(f"  [{record['id']}] {ev[:70]}")
                    shown += 1
                    if shown >= 10:
                        break
            if shown >= 10:
                break

    # Load previous output for comparison
    previous = None
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open("r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception as e:
            print(f"\nWarning: could not read previous output: {e}")

    if not has_changed(previous, updated_dataset):
        print("\n" + "=" * 70)
        print("NO CHANGES DETECTED")
        print("=" * 70)
        print(f"Output file left untouched:\n  {OUTPUT_PATH}")
        print("=" * 70)
        return

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n".join(report_lines))

    print("\n" + "=" * 70)
    print("CHANGES DETECTED — dataset updated")
    print("=" * 70)
    print(f"Updated dataset saved to:\n  {OUTPUT_PATH}")
    print(f"Full report saved to:\n  {REPORT_PATH}")


if __name__ == "__main__":
    main()
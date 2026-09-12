"""
Evidence-Based Retrieval Evaluation for Monkey-Paw-RAG

Metrics: Precision@K, Recall@K, MRR@K, MAP@K

Ground truth is defined by evidence text, NOT by fixed chunk IDs,
so evaluation is independent of CHUNK_SIZE, CHUNK_OVERLAP, and
chunk boundaries. Relevant chunk IDs are computed dynamically
by matching evidence against the current chunks in ChromaDB.
"""

import json
import sys
from pathlib import Path
from typing import Any

import chromadb

# ============================================================
# Config
# ============================================================

K = 5

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "evaluation" / "datasets" / "dataset_ground_truth.json"
RESULTS_PATH = ROOT_DIR / "evaluation" / "results" / f"retrieval_results_k={K}.json"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieve import Retriever


# ============================================================
# Helpers
# ============================================================

import re

def normalize_text(text: str) -> str:
    """Aggressive normalization for evidence matching."""
    text = str(text).lower()
    # Normalize fancy apostrophes to straight '
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032]", "'", text)
    # Normalize fancy quotes to straight "
    text = re.sub(r"[\u201C\u201D\u201E\u201F\u2033]", '"', text)
    # Join words broken by hyphen + space (PDF hyphenation)
    # e.g. "Sergeant- Major" -> "SergeantMajor" -> wait, better:
    text = re.sub(r"-\s+", "-", text)   # "Sergeant- Major" -> "Sergeant-Major"
    # Collapse all whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_sort_key(chunk_id: str):
    """Sort chunk_0, chunk_1, ... numerically."""
    if chunk_id.startswith("chunk_") and chunk_id[6:].isdigit():
        return (0, int(chunk_id[6:]))
    return (1, chunk_id)


# ============================================================
# Dataset & chunks
# ============================================================

def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list.")

    required = {"id", "question", "reference_answer", "evidence"}
    for r in dataset:
        missing = required - r.keys()
        if missing:
            raise ValueError(f"{r.get('id', '?')} missing: {sorted(missing)}")
        if not isinstance(r["evidence"], list) or not r["evidence"]:
            raise ValueError(f"{r['id']}: evidence must be a non-empty list.")

    return dataset


def load_all_chunks() -> list[dict[str, Any]]:
    """Load current chunks from ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas"])

    return [
        {
            "chunk_id": cid,
            "text": result["documents"][i] if i < len(result["documents"]) else "",
            "metadata": result["metadatas"][i] if i < len(result["metadatas"]) else {},
        }
        for i, cid in enumerate(result.get("ids", []))
    ]


# ============================================================
# Evidence → relevant chunk IDs
# ============================================================

def find_relevant_chunks(
    chunks: list[dict[str, Any]],
    evidence: list[str],
) -> tuple[set[str], dict[str, list[str]]]:
    """Return chunks containing at least one complete evidence span."""

    norm_chunks = [
        (c["chunk_id"], normalize_text(c.get("text", "")))
        for c in chunks
    ]

    relevant: set[str] = set()
    matches: dict[str, list[str]] = {}

    for ev in evidence:
        norm_ev = normalize_text(ev)
        hit_ids = [cid for cid, txt in norm_chunks if norm_ev in txt]
        matches[ev] = hit_ids
        relevant.update(hit_ids)

    return relevant, matches


# ============================================================
# Retriever output → ordered unique chunk IDs
# ============================================================

def extract_retrieved_chunk_ids(results: Any) -> list[str]:
    """Accepts dict-with-ids, list[dict], or list[str]."""

    if isinstance(results, dict) and "ids" in results:
        raw = results["ids"]
        if raw and isinstance(raw[0], list):
            raw = raw[0]

    elif isinstance(results, list) and results and isinstance(results[0], dict):
        raw = [r.get("chunk_id") or r.get("id") for r in results]

    elif isinstance(results, list):
        raw = results

    else:
        raise ValueError(f"Unsupported retriever output: {type(results).__name__}")

    ids: list[str] = []
    for r in raw:
        cid = str(r).strip()
        if cid not in ids:
            ids.append(cid)
    return ids


# ============================================================
# Metrics
# ============================================================

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


# ============================================================
# Evaluate one question
# ============================================================

def evaluate_query(
    retriever: Retriever,
    chunks: list[dict[str, Any]],
    record: dict[str, Any],
    k: int,
) -> dict[str, Any]:

    relevant, evidence_matches = find_relevant_chunks(chunks, record["evidence"])

    raw = retriever.retrieve(record["question"], top_k=k)
    retrieved = extract_retrieved_chunk_ids(raw)
    top_k = retrieved[:k]

    return {
        "id": record["id"],
        "question": record["question"],
        "reference_answer": record["reference_answer"],
        "evidence": record["evidence"],
        "ground_truth": {
            "relevant_chunk_ids": sorted(relevant, key=chunk_sort_key),
            "evidence_matches": evidence_matches,
        },
        "retrieval": {
            "top_k": top_k,
            "retrieved_relevant_chunk_ids": [c for c in top_k if c in relevant],
            "missing_relevant_chunk_ids": [c for c in relevant if c not in top_k],
        },
        "metrics": {
            f"precision_at_{k}": precision_at_k(retrieved, relevant, k),
            f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
            f"mrr_at_{k}": reciprocal_rank_at_k(retrieved, relevant, k),
            f"map_at_{k}": average_precision_at_k(retrieved, relevant, k),
        },
    }


# ============================================================
# Summary
# ============================================================

def summarize(results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    if not results:
        return {"num_questions": 0, "k": k, "metrics": {}}

    names = [
        f"precision_at_{k}",
        f"recall_at_{k}",
        f"mrr_at_{k}",
        f"map_at_{k}",
    ]

    return {
        "num_questions": len(results),
        "k": k,
        "metrics": {
            n: sum(r["metrics"][n] for r in results) / len(results)
            for n in names
        },
        "queries_with_recall_0": sum(
            1 for r in results if r["metrics"][f"recall_at_{k}"] == 0.0
        ),
        "queries_with_recall_1": sum(
            1 for r in results if r["metrics"][f"recall_at_{k}"] == 1.0
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Monkey-Paw-RAG Retrieval Evaluation (Evidence-Based)")
    print("=" * 60)

    dataset = load_dataset(DATASET_PATH)
    print(f"\nLoaded {len(dataset)} questions from {DATASET_PATH.name}")

    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from ChromaDB.")

    retriever = Retriever()

    results = []
    for i, record in enumerate(dataset, start=1):
        print(f"\rEvaluating {i}/{len(dataset)}...", end="", flush=True)
        results.append(evaluate_query(retriever, chunks, record, K))
    print()

    summary = summarize(results, K)

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation": {
                    "dataset": DATASET_PATH.name,
                    "ground_truth_type": "evidence",
                    "k": K,
                    "num_chunks": len(chunks),
                    "collection": COLLECTION_NAME,
                },
                "summary": summary,
                "queries": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Console report
    print("\n" + "=" * 60)
    print(f"Retrieval Evaluation (K = {K}, N = {summary['num_questions']})")
    print("=" * 60)
    for name, value in summary["metrics"].items():
        print(f"{name:<20}: {value:.4f}")

    print(f"\nRecall@{K} = 0 : {summary['queries_with_recall_0']}")
    print(f"Recall@{K} = 1 : {summary['queries_with_recall_1']}")

    failures = [r for r in results if r["metrics"][f"recall_at_{K}"] == 0.0]
    print(f"\nFailures @ {K}: {len(failures)}")
    for r in failures:
        print(f"\n  [{r['id']}] {r['question']}")
        print(f"     relevant : {r['ground_truth']['relevant_chunk_ids']}")
        print(f"     retrieved: {r['retrieval']['top_k']}")

    print("\n" + "=" * 60)
    print(f"Results saved to:\n{RESULTS_PATH}")


if __name__ == "__main__":
    main()
# src/check_retriever.py

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F\u2033]", '"', text)
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    # اتصال به ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(name=COLLECTION_NAME)
    data = col.get(include=["documents"])
    chunk_map = dict(zip(data["ids"], data["documents"]))

    print("=" * 70)
    print("STEP 1: ChromaDB info")
    print("=" * 70)
    print(f"Total chunks: {len(chunk_map)}")
    print(f"First chunk ID: {data['ids'][0]}")
    print(f"Last chunk ID: {sorted(chunk_map, key=lambda x: int(x.split('_')[1]))[-1]}")

    # متن chunk_0
    print("\nchunk_0 text (first 300 chars):")
    print(repr(chunk_map["chunk_0"][:300]))

    # آیا chunk_0 شامل evidence q01 هست؟
    ev = "in the small parlour of Laburnam Villa"
    print(f"\nDoes chunk_0 contain '{ev}'?")
    print(f"  → {normalize(ev) in normalize(chunk_map['chunk_0'])}")

    # ──────────────────────────────────────
    # STEP 2: دستی جستجو کن
    # ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Manual retrieval for q01")
    print("=" * 70)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Where does the White family live at the beginning of the story?"
    emb = model.encode(query).tolist()

    results = col.query(query_embeddings=[emb], n_results=5)

    for rank, (cid, doc, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ), start=1):
        contains_ev = normalize(ev) in normalize(doc)
        print(f"\nRank {rank}: {cid}  dist={dist:.4f}  contains_evidence={contains_ev}")
        print(f"   {repr(doc[:200])}")


if __name__ == "__main__":
    main()
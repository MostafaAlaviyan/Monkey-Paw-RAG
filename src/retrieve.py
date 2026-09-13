from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# =========================
# Retriever
# =========================

class Retriever:

    def __init__(
        self,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    ):
        self.embedding_model = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=str(chroma_path)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "distances": results["distances"][0],
        }


# =========================
# Test
# =========================

if __name__ == "__main__":

    retriever = Retriever()

    query = "What was the first wish?"

    results = retriever.retrieve(query, top_k=TOP_K)

    for rank, (chunk_id, document, distance) in enumerate(
        zip(
            results["ids"],
            results["documents"],
            results["distances"],
        ),
        start=1,
    ):
        print(f"\nRank {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Distance: {distance}")
        print(f"Text: {document[:300]}...")
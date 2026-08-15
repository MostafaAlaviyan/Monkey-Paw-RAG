import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

class Retriever:

    def __init__(self):

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ):
        """Retrieve the most relevant chunks."""

        query_embedding = self.embedding_model.encode(
            query
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        documents = results["documents"][0]

        return documents


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = Retriever()

    question = input(
        "Enter your question: "
    )

    results = retriever.retrieve(question)

    print("\nRetrieved context:\n")

    for i, document in enumerate(results, start=1):

        print("=" * 60)

        print(f"Chunk {i}")

        print("=" * 60)

        print(document)
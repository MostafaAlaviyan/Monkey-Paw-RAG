from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / "data" / "The-Monkeys-Paw.pdf"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "Monkey_Paw"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ---------------------------------------------------------
# Store in ChromaDB
# ---------------------------------------------------------

def store_chunks(chunks):
    """Create embeddings and store chunks in ChromaDB."""

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Avoid duplicating documents if ingest.py is run again.
    existing = collection.count()

    if existing > 0:
        print(
            f"Collection already contains {existing} documents."
        )
        print("Skipping ingestion.")

        return

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": PDF_PATH.name,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(
        f"Successfully stored {len(chunks)} chunks."
    )

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Reading PDF...")

    text = extract_text(PDF_PATH)

    print(
        f"Extracted {len(text)} characters."
    )

    print("Creating chunks...")

    chunks = create_chunks(text)

    print(
        f"Created {len(chunks)} chunks."
    )

    store_chunks(chunks)

    print("Ingestion completed.")

if __name__ == "__main__":
    main()

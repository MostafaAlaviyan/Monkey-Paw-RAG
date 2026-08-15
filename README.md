# The Monkey's Paw — Local RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about *The Monkey's Paw* using **ChromaDB**, **Sentence Transformers**, and **Llama 3.1** running locally through **Ollama**.

The project is designed as a simple, transparent implementation of a RAG pipeline, with document ingestion, vector retrieval, and LLM-based answer generation separated into independent modules.

---

## 🧠 RAG Architecture

```text
                 The Monkey's Paw PDF
                         │
                         ▼
                  Text Extraction
                         │
                         ▼
                      Chunking
                  1000 chars / 200 overlap
                         │
                         ▼
                Sentence Transformers
                 all-MiniLM-L6-v2
                         │
                         ▼
                     ChromaDB
                   Vector Store
                         │
                         │
                  User Question
                         │
                         ▼
                  Query Embedding
                         │
                         ▼
                  Similarity Search
                       Top-K=3
                         │
                         ▼
                Relevant Text Chunks
                         │
                         ▼
                  qwen2.5:0.5b
                      Ollama
                         │
                         ▼
                       Answer
```

---

## ✨ Features

* 📄 PDF document ingestion
* ✂️ Overlapping text chunking
* 🔢 Semantic embeddings with Sentence Transformers
* 🗄️ Persistent vector storage using ChromaDB
* 🔎 Top-K semantic retrieval
* 🤖 Local LLM inference with qwen2.5:0.5b
* 🦙 Ollama integration
* 🔒 Answers generated using only the retrieved story context
* 💻 Simple command-line interface
* 🧩 Modular project structure

---

## 🛠️ Technologies

* **Python**
* **PyPDF**
* **Sentence Transformers**
* **ChromaDB**
* **Ollama**
* **qwen2.5:0.5b**

### Models

**Embedding model:**

```text
all-MiniLM-L6-v2
```

**Generation model:**

```text
qwen2.5:0.5b
```

---

## 📁 Project Structure

```text
Monkey-Paw-RAG/
│
├── data/
│   └── The-Monkeys-Paw.pdf
│
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── chat.py
│   └── main.py
│
├── chroma_db/
│   └── Vector database files
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

### Source Files

| File              | Description                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `src/ingest.py`   | Extracts text from the PDF, creates chunks, generates embeddings, and stores them in ChromaDB. |
| `src/retrieve.py` | Converts user questions into embeddings and retrieves the most relevant chunks.                |
| `src/chat.py`     | Builds the prompt and sends the retrieved context to Llama 3.1 through Ollama.                 |
| `src/main.py`     | Runs the interactive RAG chatbot.                                                              |

---

## ⚙️ How It Works

### 1. Document Ingestion

`ingest.py` reads the PDF using `PyPDF`.

The extracted text is divided into overlapping chunks:

```text
Chunk size: 1000 characters
Overlap:    200 characters
```

### 2. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### 3. Vector Storage

The generated embeddings and their corresponding text chunks are stored in a persistent **ChromaDB** collection.

### 4. Retrieval

When the user asks a question, the question is converted into an embedding.

ChromaDB then performs a similarity search and returns the **top 3 most relevant chunks**.

### 5. Answer Generation

The retrieved chunks are provided to **qwen2.5:0.5b** through Ollama as context.

The model is instructed to answer using only the provided context and not rely on outside knowledge.

---
## 📖 Required Story Data

This project uses **The Monkey's Paw** by W. W. Jacobs as its source document.

Before running the ingestion script, download the story as a PDF and place it in the following location:

```text
data/The-Monkeys-Paw.pdf
```

The expected project structure is:

```text
Monkey-Paw-RAG/
│
├── data/
│   └── The-Monkeys-Paw.pdf
│
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── chat.py
│   └── main.py
│
└── ...
```

> **Note:** Make sure you have the right to download and redistribute the particular PDF version you use. If the PDF is not suitable for redistribution, do not commit it to the repository; download it separately and place it in the `data/` directory before running the project.


## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Monkey-Paw-RAG
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama for your operating system and make sure it is available from the terminal.

Then download the Llama 3.1 model:

```bash
ollama pull qwen2.5:0.5b
```

You can verify the model with:

```bash
ollama list
```

---

## ▶️ Usage

### Step 1 — Build the Vector Database

Run the ingestion script from the project root:

```bash
python src/ingest.py
```

This will:

1. Read the PDF.
2. Extract its text.
3. Create overlapping chunks.
4. Generate embeddings.
5. Store the embeddings in ChromaDB.

After successful ingestion, the `chroma_db/` directory will contain the vector database.

### Step 2 — Start the RAG Chatbot

Run:

```bash
python src/main.py
```

You should see:

```text
============================================================
        The Monkey's Paw - RAG Chatbot
============================================================

RAG chatbot is ready.
Type 'exit' to quit.
```

You can then ask questions about the story:

```text
You: What was Mr. White's first wish?

Searching the story...
Generating answer...

Assistant:
...
```

Type:

```text
exit
```

to close the application.

---

## 🔄 RAG Pipeline

The complete execution flow is:

```text
PDF
 │
 ├──► Text Extraction
 │
 ├──► Chunking
 │
 ├──► Embedding
 │
 ▼
ChromaDB
 │
 │
 └──────────────┐
                │
          User Question
                │
                ▼
         Query Embedding
                │
                ▼
        Similarity Search
                │
                ▼
         Top 3 Chunks
                │
                ▼
         Context + Question
                │
                ▼
         qwen2.5:0.5b
                │
                ▼
             Answer
```

---

## 📌 Current Limitations

This is a simple RAG implementation intended primarily for learning and experimentation.

Currently, the project does not include:

* Reranking
* Hybrid search
* Similarity-score thresholding
* Retrieval evaluation
* Conversational memory
* Multi-document retrieval
* Web interface
* LangChain

These can be added in future versions.

---

## 🔮 Future Improvements

Possible improvements include:

* [ ] Improve the chunking strategy
* [ ] Add similarity-score filtering
* [ ] Add retrieval evaluation
* [ ] Experiment with different embedding models
* [ ] Add a reranking stage
* [ ] Add conversational memory
* [ ] Support multiple documents
* [ ] Build a web interface
* [ ] Reimplement the pipeline using LangChain
* [ ] Extend the system to support voice interaction
* [ ] Experiment with Agentic RAG

---

## 📚 Learning Goals

This project focuses on understanding the fundamental components of a RAG system without hiding the implementation behind a high-level framework.

The main concepts demonstrated are:

```text
Document Loading
       ↓
Text Chunking
       ↓
Embedding
       ↓
Vector Database
       ↓
Semantic Retrieval
       ↓
Context Construction
       ↓
Local LLM
       ↓
Generated Answer
```

The project can therefore serve as a foundation for experimenting with more advanced RAG architectures.

---


## 👤 Author

**Seyed Mostafa Alaviyan Shahri**

Built as a learning and portfolio project focused on **RAG, LLMs, vector databases, and local AI inference**.

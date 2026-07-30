# ACM-ICAIF-24-FinanceRAG

Build RAG systems to analyze financial documents and answer queries using textual and tabular data, for the [ICAIF'24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge).

## Pipeline

1. **Download** — pull the competition dataset from Kaggle (`src/donwnload_data.py`).
2. **Load** — read the JSONL corpora (ConvFinQA, FinanceBench, FinDER, FinQA, FinQABench, MultiHiertt, TAT-QA) into memory (`src/load_data.py`).
3. **Preprocess** — normalize unicode/whitespace, strip boilerplate, and redact PII from document text (`src/preprocess_data.py`).
4. **Embed** — chunk documents with `RecursiveCharacterTextSplitter` and embed them with `sentence-transformers/all-MiniLM-L6-v2` via `light_embed` (`src/embedding.py`).
5. **Index** — store embeddings in a local FAISS (HNSW) index with pickled metadata (`src/vector_store.py`).
6. **Search** — run similarity search over the index and summarize retrieved context with a Groq-hosted LLM (`llama-3.1-8b-instant`) via `langchain_groq` (`src/search.py`).

## Project structure

```
src/
  donwnload_data.py    # Kaggle dataset download
  load_data.py          # JSONL corpus loading
  preprocess_data.py    # text cleaning / PII redaction
  embedding.py           # chunking + embedding pipeline
  vector_store.py        # FAISS index build/query
  search.py              # RAG query + Groq summarization
data/                   # symlink to the downloaded Kaggle dataset (gitignored)
faiss_store/             # persisted FAISS index + metadata (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu kagglehub langchain-community langchain-groq light_embed python-dotenv
```

Create a `.env` file in the project root with:

```
KAGGLE_API_TOKEN=<your kaggle token>
GROQ_API_KEY=<your groq api key>
```

## Usage

```bash
# 1. Download the competition data (requires Kaggle auth)
python src/donwnload_data.py

# 2. Build the FAISS vector store from the corpora
python src/vector_store.py

# 3. Run a RAG query
python src/search.py
```

## Data

Dataset provided by the [ICAIF'24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge) on Kaggle.

## License

MIT — see [LICENSE](LICENSE).

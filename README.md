#  Biodiversity RAG Assistant

This project is a **Retrieval-Augmented Generation (RAG)** pipeline tailored for ESG and biodiversity-focused investment research. It uses **ChromaDB** as a vector store and **OpenAI GPT models** to perform semantic document search and answer investor-focused questions using natural language.

---

##  Description

The assistant is capable of:
- Ingesting ESG/Biodiversity PDFs
- Chunking and embedding them using `sentence-transformers`
- Indexing chunks in ChromaDB
- Answering queries using `gpt-4-turbo` or other LLMs with source-based answers

---

##  Tech Stack

- Python 3.10+
- Jupyter Notebooks
- SentenceTransformers
- ChromaDB
- OpenAI API
- dotenv (.env) config

---

##  How to Run

1. Clone the repo and create the environment:
```bash
git clone https://github.com/yourname/biodiversity-rag-nlp.git
cd biodiversity-rag-nlp
conda env create -f environment.yml
conda activate bio-rag
```

2. Add your OpenAI key to a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

3. Run notebooks in order:
- `01_load_documents.ipynb`: Load and chunk PDFs
- `02_build_vector_index.ipynb`: Generate embeddings, store in Chroma
- `03_answer_with_gpt.ipynb`: Query GPT-4/GPT-3.5 with citations

---

##  Project Structure

```
biodiversity-rag-nlp/
│
├── notebooks/
│   ├── 01_load_documents.ipynb
│   ├── 02_build_vector_index.ipynb
│   └── 03_answer_with_gpt.ipynb
│
├── outputs/
│   └── flattened_docs.pkl
│
├── data/
│   └── vector_db/            # Chroma vector store (ignored by git)
│
├── .env                      # Your OpenAI Key (not tracked)
├── .gitignore
├── environment.yml
├── README.md
```

---

##  Status

- [x] Document Ingestion & Preprocessing
- [x] Embedding & Indexing in ChromaDB
- [x] GPT-4 Q&A with citations
- [ ] Multi-model comparison (coming next!)

---

##  Author

Francisco Salazar — [franquant@gmail.com](mailto:franquant@gmail.com)


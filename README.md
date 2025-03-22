# Biodiversity RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline tailored for ESG and biodiversity-focused investment research. Built with ChromaDB and OpenAI GPT models, this system enables semantic search and question answering with transparent source citations from biodiversity documents.

## Key Capabilities

- Ingest and process ESG/Biodiversity reports (PDFs)
- Embed document chunks with `sentence-transformers`
- Index into a persistent ChromaDB vector store
- Query with GPT-4 or GPT-3.5, returning answers grounded in document context
- Answers include similarity-based citations for traceability

## Tech Stack

- Python 3.10+
- Jupyter Notebooks
- SentenceTransformers
- ChromaDB
- OpenAI API
- dotenv for environment config

## Getting Started

1. Clone the repo and set up the environment:

   ```bash
   git clone https://github.com/FranQuant/biodiversity-rag-nlp.git
   cd biodiversity-rag-nlp
   conda env create -f environment.yml
   conda activate bio-rag
   ```

2. Add your OpenAI key to `.env`:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Run the notebooks in order:

   - `01_load_documents.ipynb` → Chunk and flatten the PDFs
   - `02_build_vector_index.ipynb` → Generate embeddings and store them
   - `03_answer_with_gpt.ipynb` → Ask questions with GPT and get cited answers

## Folder Structure

```
biodiversity-rag-nlp/
├── notebooks/
│   ├── 01_load_documents.ipynb
│   ├── 02_build_vector_index.ipynb
│   └── 03_answer_with_gpt.ipynb
├── data/
│   └── vector_db/            (excluded via .gitignore)
├── outputs/
│   └── flattened_docs.pkl
├── .env                      (excluded)
├── .gitignore
├── environment.yml
├── README.md
```

## Project Status

- Document ingestion & chunking: Complete
- Embedding & persistent vector indexing: Complete
- GPT-4 question answering with citations: Complete
- Multi-model comparison: Coming soon

## Author

Francisco Salazar  
Email: franquant@gmail.com  
GitHub: https://github.com/FranQuant

---

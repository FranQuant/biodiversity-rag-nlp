"""
📚 01_load_documents.py - Multi-Source ESG/Biodiversity Ingestion & Chunking

This script loads, merges, and chunks documents from multiple sources:
- 📄 Local PDFs
- 🌐 Web URLs
- 📊 CSV Files

It processes and prepares documents for vector embedding in `02_build_vector_index.py`.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PandasCSVReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser

# Load environment variables
load_dotenv()

# Function: Load documents from multiple sources
def load_all_documents(pdf_path: Path, url_list: list = None, csv_paths: list = None):
    """
    Load ESG/Biodiversity documents from PDFs, URLs, and CSVs.

    Args:
        pdf_path (Path): Directory containing PDF files.
        url_list (list, optional): List of URLs to fetch and parse.
        csv_paths (list, optional): List of CSV file paths to load.

    Returns:
        List[Document]: Unified list of LlamaIndex Document objects.
    """
    all_documents = []

    # Load PDFs
    pdf_docs = SimpleDirectoryReader(
        input_dir=pdf_path, recursive=False, required_exts=[".pdf"],
        file_extractor={'pdf': 'pymupdf'}, filename_as_id=True
    ).load_data()
    print(f"✅ Loaded {len(pdf_docs)} PDF docs.")
    all_documents.extend(pdf_docs)

    # Load URLs
    if url_list:
        web_docs = SimpleWebPageReader().load_data(urls=url_list)
        print(f"✅ Loaded {len(web_docs)} web pages.")
        all_documents.extend(web_docs)

    # Load CSVs
    if csv_paths:
        csv_reader = PandasCSVReader()
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"⚠️ File not found: {csv_path}")
                continue
            csv_docs = csv_reader.load_data(Path(csv_path))
            print(f"✅ Loaded {len(csv_docs)} docs from {csv_path}")
            all_documents.extend(csv_docs)
    
    return all_documents

# Define paths and sources
pdf_path = Path("../data/raw")
url_list = [
    "https://www.unep.org/resources/report/state-finance-biodiversity",
    "https://www.cdp.net/en/research/global-reports/global-biodiversity-report-2023",
    "https://ipbes.net/global-assessment",
    "https://wwf.panda.org/discover/knowledge_hub/all_publications/living_planet_report_2022/",
    "https://environment.ec.europa.eu/strategy/biodiversity-strategy-2030_en"
]
csv_paths = ["10k_biodiversity_scores.csv"]

# Load documents
documents = load_all_documents(pdf_path, url_list, csv_paths)
print(f"📚 Total documents loaded: {len(documents)}")

# Merge PDFs by filename; keep web and CSV docs intact
merged_documents = {}
for doc in documents:
    file_name = doc.metadata.get("file_name")
    if file_name:
        if file_name not in merged_documents:
            merged_documents[file_name] = Document(text=doc.text, metadata=doc.metadata)
        else:
            merged_documents[file_name] = Document(
                text=merged_documents[file_name].text + "\n" + doc.text,
                metadata=merged_documents[file_name].metadata
            )
    else:
        fallback_key = doc.metadata.get("source", f"source_{len(merged_documents)}")
        merged_documents[fallback_key] = doc

documents = list(merged_documents.values())
print(f"✅ Merged documents total: {len(documents)}")

# Apply controlled chunking
parser = SimpleNodeParser.from_defaults(
    chunk_size=1024, chunk_overlap=200
)
chunked_documents = [parser.get_nodes_from_documents([doc]) for doc in documents]
flattened_docs = [node for sublist in chunked_documents for node in sublist]
print(f"✅ Chunked into {len(flattened_docs)} total chunks.")

# Visualize chunk distribution
source_labels = [node.metadata.get("file_name") or node.metadata.get("source") or "Unknown" for node in flattened_docs]
chunk_counts = Counter(source_labels)
top_sources = chunk_counts.most_common(15)

plt.figure(figsize=(12, 6))
plt.barh(
    [k for k, _ in top_sources],
    [v for _, v in top_sources]
)
plt.xlabel("Number of Chunks")
plt.title("Distribution of Chunks per Document")
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Save flattened documents (Optional, for indexing in Notebook 2)
import pickle
with open("flattened_docs.pkl", "wb") as f:
    pickle.dump(flattened_docs, f)
print("✅ Saved flattened_docs to disk.")

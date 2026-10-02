# Hospital Knowledge Base RAG

A professional starter knowledge base for a Retrieval-Augmented Generation (RAG) assistant.

## Disclaimer
All hospital content is fictional demonstration data. It must not be used for real clinical decisions, patient care, diagnosis, treatment, or operational decisions.

## Knowledge base
Fictional PDF documents organized by department:
- Emergency
- Cardiology
- Pharmacy
- Outpatient
- Infection Control
- Patient Services

## Ingestion
PDF -> pypdf extraction -> cleaning -> overlapping chunks -> Sentence Transformers embeddings -> normalized vectors -> FAISS IndexFlatIP -> metadata.

## Run
pip install -r requirements.txt
python ingest.py

Generated:
- faiss_index/index.faiss
- faiss_index/metadata.pkl

## Embedding model
sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).

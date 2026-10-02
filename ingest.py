from pathlib import Path
import pickle
import re
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("hospital_knowledge_base")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()

def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return clean_text("\n".join((page.extract_text() or "") for page in reader.pages))

def split_text(text: str):
    if not text:
        return []
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def discover_pdfs():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Knowledge base not found: {DATA_DIR.resolve()}")
    return sorted(DATA_DIR.rglob("*.pdf"))

def build_chunks(pdf_files):
    records = []
    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path)
        if not text:
            print(f"Skipped empty PDF: {pdf_path}")
            continue
        relative_path = pdf_path.relative_to(DATA_DIR)
        department = relative_path.parts[0] if len(relative_path.parts) > 1 else "General"
        for chunk_id, chunk in enumerate(split_text(text)):
            records.append({"text": chunk, "source": str(relative_path),
                            "file_name": pdf_path.name, "department": department,
                            "chunk_id": chunk_id})
        print(f"Processed: {relative_path}")
    return records

def main():
    pdf_files = discover_pdfs()
    if not pdf_files:
        raise RuntimeError("No PDF files found.")
    records = build_chunks(pdf_files)
    if not records:
        raise RuntimeError("No text chunks were created.")
    print(f"Documents: {len(pdf_files)} | Chunks: {len(records)}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([r["text"] for r in records], batch_size=32,
                              show_progress_bar=True, convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
    if embeddings.shape[0] != len(records):
        raise RuntimeError("Embedding count does not match metadata count.")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as file:
        pickle.dump(records, file)
    print(f"Done. Vectors: {index.ntotal}, Dimension: {index.d}")

if __name__ == "__main__":
    main()

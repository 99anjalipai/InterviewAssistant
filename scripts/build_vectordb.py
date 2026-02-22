import os
import glob
import json
from typing import List, Dict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_DIR = "candidate_docs"
OUT_DIR = os.path.join("stores", "candidate_faiss")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def infer_doc_type(filename: str) -> str:
    name = os.path.basename(filename).lower()
    if name.startswith("resume"):
        return "resume"
    if name.startswith("work_experience") or "ltimindtree" in name:
        return "work_exp"
    if name.startswith("project"):
        return "project"
    if "star" in name:
        return "star_story"
    if "preference" in name or "writing" in name:
        return "writing_prefs"
    return "candidate_misc"


def load_candidate_files() -> List[str]:
    paths = sorted(glob.glob(os.path.join(CANDIDATE_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in '{CANDIDATE_DIR}'. "
            f"Create the folder and add your candidate docs."
        )
    return paths


def build_documents(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        meta: Dict = {
            "doc_type": infer_doc_type(p),
            "source_file": os.path.basename(p),
            "source_id": os.path.splitext(os.path.basename(p))[0],
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def main():
    paths = load_candidate_files()
    raw_docs = build_documents(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    os.makedirs(OUT_DIR, exist_ok=True)
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(OUT_DIR)

    manifest = {
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_source_files": len(paths),
        "num_chunks": len(chunks),
        "files": [os.path.basename(p) for p in paths],
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Candidate FAISS index built successfully")
    print(f"   Source files: {len(paths)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
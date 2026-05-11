#!/usr/bin/env python3
"""
data_update.py — Idempotent RAG ingestion pipeline.

Pipeline:
  1. Clean raw files (data/raw/) → data/processed/
  2. Chunk cleaned text
  3. Embed with sentence-transformers
  4. Write to pgvector (incremental or full rebuild)

Usage:
  python data_update.py            # incremental update
  python data_update.py --rebuild  # full rebuild
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STATE_FILE = PROCESSED_DIR / ".file_state.json"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")


# ---------------------------------------------------------------------------
# Stage 1: Clean raw files → data/processed/
# ---------------------------------------------------------------------------


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")  # strip NUL bytes (PDF artefact)
    text = re.sub(r"<[^>]+>", "", text)  # strip HTML tags
    text = re.sub(r"[ \t]+", " ", text)  # normalize spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    return text.strip()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def clean_raw_files(rebuild: bool) -> list[str]:
    """Returns list of source stems that were (re)processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    updated_stems = []

    raw_files = [
        p for p in RAW_DIR.iterdir() if p.suffix.lower() in (".pdf", ".md", ".txt")
    ]

    if rebuild:
        # Remove all processed files and reset state
        for f in PROCESSED_DIR.glob("*.txt"):
            f.unlink()
        state = {}

    for raw_path in raw_files:
        stem = raw_path.stem
        mtime = raw_path.stat().st_mtime
        processed_path = PROCESSED_DIR / f"{stem}.txt"

        if not rebuild and state.get(stem) == mtime and processed_path.exists():
            print(f"  [skip] {raw_path.name} (unchanged)")
            continue

        print(f"  [process] {raw_path.name}")
        try:
            raw_text = extract_text(raw_path)
            cleaned = clean_text(raw_text)
            processed_path.write_text(cleaned, encoding="utf-8")
            state[stem] = mtime
            updated_stems.append(stem)
        except Exception as e:
            print(f"  [error] {raw_path.name}: {e}", file=sys.stderr)

    save_state(state)
    return updated_stems


# ---------------------------------------------------------------------------
# Stage 2: Chunk
# ---------------------------------------------------------------------------


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def load_chunks(stems: list[str]) -> list[tuple[str, int, str]]:
    """Returns list of (source_stem, chunk_index, chunk_text)."""
    result = []
    for stem in stems:
        path = PROCESSED_DIR / f"{stem}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text)):
            result.append((stem, idx, chunk))
    return result


# ---------------------------------------------------------------------------
# Stage 3: Embed
# ---------------------------------------------------------------------------


def embed(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts, show_progress_bar=True).tolist()
    elif EMBEDDING_PROVIDER == "huggingface":
        import requests

        hf_token = os.getenv("HF_TOKEN", "")
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        resp.raise_for_status()
        return resp.json()
    elif EMBEDDING_PROVIDER == "ollama":
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return vectors
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


# ---------------------------------------------------------------------------
# Stage 4: Write to pgvector
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(384)
);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON document_chunks(source);
"""


def get_conn():
    import psycopg

    if not PGVECTOR_CONNECTION_STRING:
        print("ERROR: PGVECTOR_CONNECTION_STRING is not set in .env", file=sys.stderr)
        sys.exit(1)
    return psycopg.connect(PGVECTOR_CONNECTION_STRING)


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def delete_sources(conn, stems: list[str]):
    if not stems:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "DELETE FROM document_chunks WHERE source = %s",
            [(s,) for s in stems],
        )
    conn.commit()


def truncate_table(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE document_chunks")
    conn.commit()


def insert_chunks(conn, rows: list[tuple[str, int, str, list[float]]]):
    from pgvector.psycopg import register_vector

    register_vector(conn)
    import numpy as np

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO document_chunks (source, chunk_index, text, embedding) VALUES (%s, %s, %s, %s)",
            [(source, idx, text, np.array(vec)) for source, idx, text, vec in rows],
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG data ingestion pipeline")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear processed data and rebuild entire index",
    )
    args = parser.parse_args()

    if not PGVECTOR_CONNECTION_STRING:
        print("ERROR: PGVECTOR_CONNECTION_STRING is not set in .env", file=sys.stderr)
        sys.exit(1)

    print("=== Stage 1: Clean raw files ===")
    updated_stems = clean_raw_files(rebuild=args.rebuild)

    if not updated_stems:
        print("No files changed. Nothing to do.")
        return

    print(f"\n=== Stage 2: Chunk ({len(updated_stems)} file(s)) ===")
    chunk_records = load_chunks(updated_stems)
    print(f"  {len(chunk_records)} chunks total")

    print("\n=== Stage 3: Embed ===")
    texts = [c[2] for c in chunk_records]
    vectors = embed(texts)

    print("\n=== Stage 4: Write to pgvector ===")
    rows = [
        (src, idx, txt, vec) for (src, idx, txt), vec in zip(chunk_records, vectors)
    ]
    with get_conn() as conn:
        ensure_schema(conn)
        if args.rebuild:
            truncate_table(conn)
        else:
            delete_sources(conn, updated_stems)
        insert_chunks(conn, rows)
    print(f"  Wrote {len(rows)} chunks to database.")

    print("\nDone.")


if __name__ == "__main__":
    main()

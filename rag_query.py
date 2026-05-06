#!/usr/bin/env python3
"""
rag_query.py — RAG query interface with reranking.

Pipeline per query:
  1. Embed query
  2. Retrieve top-k*4 candidates from pgvector (cosine similarity)
  3. Rerank with cross-encoder, keep top-k
  4. Assemble prompt with context + history
  5. Call LLM via LiteLLM
  6. Display answer + sources

Usage:
  python rag_query.py                              # interactive mode
  python rag_query.py --query "your question"
  python rag_query.py --query "..." --top-k 3 --model gemini-2.5-flash
"""

import argparse
import os
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = "qwen-local"
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not in the context, say so clearly."
)


# ---------------------------------------------------------------------------
# Step 1: Embed query
# ---------------------------------------------------------------------------


def embed_query(text: str) -> list[float]:
    if EMBEDDING_PROVIDER == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode([text])[0].tolist()
    elif EMBEDDING_PROVIDER == "huggingface":
        import requests

        hf_token = os.getenv("HF_TOKEN", "")
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": [text], "options": {"wait_for_model": True}},
        )
        resp.raise_for_status()
        return resp.json()[0]
    elif EMBEDDING_PROVIDER == "ollama":
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


# ---------------------------------------------------------------------------
# Step 2: Retrieve candidates from pgvector
# ---------------------------------------------------------------------------


def get_conn():
    import psycopg

    return psycopg.connect(PGVECTOR_CONNECTION_STRING)


def retrieve_chunks(query_vec: list[float], candidates: int) -> list[dict]:
    from pgvector.psycopg import register_vector

    vec = np.array(query_vec)
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk_index, text, 1 - (embedding <=> %s) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, vec, candidates),
            )
            rows = cur.fetchall()
    return [
        {"source": r[0], "chunk_index": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Step 3: Rerank with cross-encoder
# ---------------------------------------------------------------------------


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Step 4: Build prompt messages
# ---------------------------------------------------------------------------


def build_messages(
    history: list[dict],
    question: str,
    chunks: list[dict],
) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] (source: {chunk['source']}, chunk #{chunk['chunk_index']})\n{chunk['text']}"
        )
    context_block = "\n\n".join(context_parts)

    user_content = f"Context:\n{context_block}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Step 5: Call LLM via LiteLLM
# ---------------------------------------------------------------------------


def call_llm(messages: list[dict], model: str) -> str:
    import litellm

    # Gemini models use OPENAI_API_KEY (a separate key with Gemini access on the proxy).
    # All other models use LITELLM_API_KEY routed through the OpenAI-compatible proxy.
    if model.startswith("gemini"):
        api_key = OPENAI_API_KEY
    else:
        api_key = LITELLM_API_KEY
    routed_model = f"openai/{model}" if LITELLM_BASE_URL else model
    response = litellm.completion(
        model=routed_model,
        messages=messages,
        api_base=LITELLM_BASE_URL,
        api_key=api_key,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 6: Display sources
# ---------------------------------------------------------------------------


def show_sources(chunks: list[dict]):
    parts = [
        f"[{i}] {c['source']} chunk #{c['chunk_index']}"
        for i, c in enumerate(chunks, 1)
    ]
    print(f"\nSources: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


def run_query(question: str, top_k: int, model: str, history: list[dict]) -> str:
    print("Embedding query...", end=" ", flush=True)
    vec = embed_query(question)
    print("done.")

    print(f"Retrieving {top_k * 4} candidates...", end=" ", flush=True)
    candidates = retrieve_chunks(vec, top_k * 4)
    print("done.")

    print(f"Reranking to top {top_k}...", end=" ", flush=True)
    chunks = rerank(question, candidates, top_k)
    print("done.")

    messages = build_messages(history, question, chunks)

    print("Generating answer...\n")
    answer = call_llm(messages, model)
    print(answer)
    show_sources(chunks)
    return answer


def single_query_mode(query: str, top_k: int, model: str):
    run_query(query, top_k, model, history=[])


def interactive_mode(top_k: int, model: str):
    print(f"RAG Query Interface (model: {model}, top-k: {top_k})")
    print("Type your question. Press Ctrl+C or Ctrl+D to exit.\n")

    history: list[dict] = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        answer = run_query(question, top_k, model, history)

        # Keep last 3 turns (6 messages: user+assistant pairs)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        history = history[-(3 * 2) :]
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG query interface")
    parser.add_argument(
        "--query", "-q", type=str, help="Single query (non-interactive)"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL}). Examples: gpt-oss:20b, gemini-2.5-flash",
    )
    args = parser.parse_args()

    if not PGVECTOR_CONNECTION_STRING:
        print("ERROR: PGVECTOR_CONNECTION_STRING is not set in .env", file=sys.stderr)
        sys.exit(1)

    if args.query:
        single_query_mode(args.query, args.top_k, args.model)
    else:
        interactive_mode(args.top_k, args.model)


if __name__ == "__main__":
    main()

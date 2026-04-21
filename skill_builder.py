#!/usr/bin/env python3
"""
skill_builder.py — Systematically extract knowledge from the RAG knowledge base
and generate a skill.md file in Agent Skill format.

Pipeline:
  1. Topic Scanning: run preset global questions through the RAG pipeline
  2. Knowledge Integration: synthesize all answers via LLM
  3. Skill.md Generation: write output in Agent Skill template format

Usage:
  python skill_builder.py
  python skill_builder.py --output skill.md --model gemini-2.5-flash
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Reuse all RAG helpers from rag_query — no duplication
from rag_query import embed_query, retrieve_chunks, rerank, call_llm

PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_OUTPUT = "skill.md"
DEFAULT_TOP_K = 5

PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Preset global questions — one per target section
# ---------------------------------------------------------------------------

GLOBAL_QUESTIONS = [
    (
        "core_concepts",
        "What are the main concepts and subtopics covered in this knowledge base? "
        "List 5–15 of the most important concepts, each with a 1–2 sentence explanation.",
    ),
    (
        "key_trends",
        "What are the most important current research directions and trends in this field? "
        "List 3–10 active development directions or emerging topics.",
    ),
    (
        "key_entities",
        "Who are the main authors, research groups, institutions, and what are the key tools, "
        "frameworks, and datasets mentioned across these documents? Group them by category.",
    ),
    (
        "methodology",
        "What methods, workflows, or best practices are widely accepted or frequently recommended "
        "in this field according to the documents?",
    ),
    (
        "gaps",
        "What are the known limitations, open challenges, knowledge gaps, or topics that are "
        "underrepresented or explicitly noted as future work in these documents?",
    ),
    (
        "example_qa",
        "Generate 3–5 representative question-and-answer pairs that demonstrate the kinds of "
        "questions this knowledge base can answer well.",
    ),
]


# ---------------------------------------------------------------------------
# Phase 1: Topic scanning — run each question through full RAG pipeline
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not in the context, say so clearly."
)


def build_rag_messages(question: str, chunks: list[dict]) -> list[dict]:
    context_parts = [
        f"[{i}] (source: {c['source']}, chunk #{c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    context_block = "\n\n".join(context_parts)
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]


def ask_rag(question: str, top_k: int, model: str) -> str:
    print(f"  Embedding...", end=" ", flush=True)
    vec = embed_query(question)
    print("Retrieving...", end=" ", flush=True)
    candidates = retrieve_chunks(vec, top_k * 4)
    print("Reranking...", end=" ", flush=True)
    chunks = rerank(question, candidates, top_k)
    print("Generating...", end=" ", flush=True)
    messages = build_rag_messages(question, chunks)
    answer = call_llm(messages, model)
    print("done.")
    return answer


def scan_topics(top_k: int, model: str) -> dict[str, str]:
    results = {}
    for key, question in GLOBAL_QUESTIONS:
        print(f"\n[Q: {key}]")
        print(f"  \"{question[:80]}...\"" if len(question) > 80 else f"  \"{question}\"")
        results[key] = ask_rag(question, top_k, model)
    return results


# ---------------------------------------------------------------------------
# Phase 2: Knowledge integration — synthesize into overview paragraph
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a knowledge synthesis expert. "
    "Given structured Q&A results from a RAG knowledge base, write a concise overview paragraph "
    "(under 200 words) summarizing the core knowledge domain, scope, and capabilities of the knowledge base."
)


def synthesize_overview(qa_results: dict[str, str], model: str) -> str:
    qa_text = "\n\n".join(
        f"**{key}**:\n{answer}" for key, answer in qa_results.items()
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here are the Q&A results from the knowledge base:\n\n{qa_text}"},
    ]
    print("\n[Synthesizing overview...]", end=" ", flush=True)
    overview = call_llm(messages, model)
    print("done.")
    return overview


# ---------------------------------------------------------------------------
# Phase 3: Skill.md generation
# ---------------------------------------------------------------------------

def count_sources() -> int:
    if not PROCESSED_DIR.exists():
        return 0
    return len(list(PROCESSED_DIR.glob("*.txt")))


def list_sources() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.txt"))


def render_skill_md(qa_results: dict[str, str], overview: str) -> str:
    today = date.today().isoformat()
    n_sources = count_sources()
    sources = list_sources()
    source_list = "\n".join(f"- {s}" for s in sources) if sources else "- (none found)"

    return f"""# Skill: Decentralized Federated Learning Research Assistant

## Metadata
- **知識領域**：Decentralized Federated Learning / Distributed Machine Learning
- **資料來源數量**：{n_sources} 份文件
- **最後更新時間**：{today}
- **適用 Agent 類型**：研究助手 / 技術顧問 / 領域問答機器人

## Overview（一段話摘要）
{overview}

## Core Concepts（核心概念）
{qa_results['core_concepts']}

## Key Trends（最新趨勢）
{qa_results['key_trends']}

## Key Entities（重要實體）
{qa_results['key_entities']}

## Methodology & Best Practices（方法論與最佳實踐）
{qa_results['methodology']}

## Knowledge Gaps & Limitations（知識邊界）
{qa_results['gaps']}

## Example Q&A（代表性問答）
{qa_results['example_qa']}

## Source References（來源索引）
{source_list}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate skill.md from RAG knowledge base")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_MODEL}). Examples: gpt-oss:20b, gemini-2.5-flash")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Chunks to retrieve per question (default: {DEFAULT_TOP_K})")
    args = parser.parse_args()

    if not PGVECTOR_CONNECTION_STRING:
        print("ERROR: PGVECTOR_CONNECTION_STRING is not set in .env", file=sys.stderr)
        sys.exit(1)

    print("=== Phase 1: Topic Scanning ===")
    qa_results = scan_topics(args.top_k, args.model)

    print("\n=== Phase 2: Knowledge Integration ===")
    overview = synthesize_overview(qa_results, args.model)

    print("\n=== Phase 3: Generating Skill.md ===")
    content = render_skill_md(qa_results, overview)
    Path(args.output).write_text(content, encoding="utf-8")
    print(f"  Written to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()

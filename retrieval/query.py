"""
query.py
--------
RAG retrieval + generation layer.

Given a natural language question, this module:
  1. Embeds the question with nomic-embed-text
  2. Retrieves the top-k most relevant round chunks from ChromaDB
  3. Builds a prompt with that context
  4. Calls a local Ollama LLM to produce an actionable insight

Usage (as a module):
    from retrieval.query import ask
    answer = ask("Why am I losing pistol rounds?")

Usage (CLI):
    python retrieval/query.py "Why do I keep losing on B site?"
"""

import argparse
import textwrap
from typing import Optional

import chromadb
from chromadb.config import Settings
import ollama


# ---------------------------------------------------------------------------
# Config — change LLM_MODEL to match what you have pulled in Ollama
# ---------------------------------------------------------------------------

CHROMA_PATH = "data/chroma_db"
COLLECTION  = "cs2_rounds"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"          # or "mistral", "llama3.1:8b", etc.
TOP_K       = 8                   # how many rounds to retrieve per query


# ---------------------------------------------------------------------------
# Shared clients (lazily initialised once)
# ---------------------------------------------------------------------------

_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION)
    return _collection


# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = TOP_K,
             filters: Optional[dict] = None) -> list[dict]:
    """
    Embed the question and return the top_k most similar round chunks.

    `filters` is passed straight to ChromaDB's `where` clause, e.g.:
        {"map": "de_mirage"}
        {"winner_side": "t"}
    """
    # Embed the query
    response   = ollama.embed(model=EMBED_MODEL, input=question)
    query_vec  = response["embeddings"][0]

    collection = _get_collection()
    results    = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=filters or None,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})

    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt from retrieved context."""
    context_lines = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        context_lines.append(
            f"[Round {meta.get('round','?')} | {meta.get('map','?')} | "
            f"Winner: {meta.get('winner_side','?')}]\n{c['text']}"
        )
    context_block = "\n\n".join(context_lines)

    prompt = textwrap.dedent(f"""
        You are an expert CS2 performance coach analysing a player's match data.
        Below are the most relevant rounds retrieved from the player's match history.
        Use ONLY this data to answer the question. Be specific and actionable.
        If the data is insufficient to answer confidently, say so.

        --- MATCH DATA ---
        {context_block}
        --- END DATA ---

        Player question: {question}

        Provide a structured analysis with:
        1. What the data shows
        2. The root cause of the issue (if identifiable)
        3. Concrete steps to improve
    """).strip()

    return prompt


def ask(question: str, filters: Optional[dict] = None,
        top_k: int = TOP_K) -> str:
    """
    Full RAG pipeline: retrieve → prompt → generate.
    Returns the LLM's answer as a string.
    """
    chunks = retrieve(question, top_k=top_k, filters=filters)

    if not chunks:
        return "No relevant match data found. Make sure you have ingested demos first."

    prompt   = build_prompt(question, chunks)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},   # low temp = more factual
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ask the CS2 RAG coach a question.")
    ap.add_argument("question", help='e.g. "Why do I lose pistol rounds?"')
    ap.add_argument("--map",    default=None, help="Filter to a specific map, e.g. de_mirage")
    ap.add_argument("--top-k",  type=int, default=TOP_K)
    args = ap.parse_args()

    filters = {"map": args.map} if args.map else None

    print(f"\n🔍 Retrieving top {args.top_k} relevant rounds...\n")
    chunks = retrieve(args.question, top_k=args.top_k, filters=filters)
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        print(f"  {i}. Round {m.get('round')} | {m.get('map')} | "
              f"Winner: {m.get('winner_side')} | Score: {c['distance']:.3f}")

    print(f"\n🤖 Generating insight with {LLM_MODEL}...\n")
    answer = ask(args.question, filters=filters, top_k=args.top_k)
    print("=" * 60)
    print(answer)
    print("=" * 60)

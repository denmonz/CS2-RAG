"""
embed_and_store.py
------------------
Reads parsed round-chunk JSON files, embeds each chunk using
nomic-embed-text (via Ollama), and stores them in a local ChromaDB collection.

Usage:
    python ingest/embed_and_store.py                        # ingest all files in data/parsed/
    python ingest/embed_and_store.py --file data/parsed/match1.json
"""

import json
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
import ollama


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PATH    = "data/chroma_db"
COLLECTION     = "cs2_rounds"
EMBED_MODEL    = "nomic-embed-text"   # pulled via: ollama pull nomic-embed-text
PARSED_DIR     = "data/parsed"
BATCH_SIZE     = 32                   # embed N chunks at once to avoid OOM


# ---------------------------------------------------------------------------
# ChromaDB client (persistent, local)
# ---------------------------------------------------------------------------

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create so re-running is safe (won't duplicate)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call Ollama's embed endpoint for a batch of texts."""
    embeddings = []
    for text in texts:
        response = ollama.embed(model=EMBED_MODEL, input=text)
        embeddings.append(response["embeddings"][0])
    return embeddings


# ---------------------------------------------------------------------------
# Ingest a list of chunks
# ---------------------------------------------------------------------------

def ingest_chunks(chunks: list[dict], collection: chromadb.Collection) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    Returns the number of newly added chunks.
    """
    # Filter out chunks already in the DB (idempotent ingest)
    existing_ids = set(collection.get(ids=[c["chunk_id"] for c in chunks])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("  [ingest] All chunks already present — skipping.")
        return 0

    added = 0
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids   = [c["chunk_id"] for c in batch]
        metas = [c["metadata"] for c in batch]

        print(f"  [ingest] Embedding batch {i // BATCH_SIZE + 1} "
              f"({len(batch)} chunks)...")
        vectors = embed_texts(texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
        )
        added += len(batch)

    return added


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(file: str | None = None):
    collection = get_collection()
    print(f"[ingest] Connected to ChromaDB — collection '{COLLECTION}' "
          f"({collection.count()} existing chunks)")

    if file:
        json_files = [Path(file)]
    else:
        json_files = sorted(Path(PARSED_DIR).glob("*.json"))

    if not json_files:
        print("[ingest] No JSON files found. Run parse_demo.py first.")
        return

    total_added = 0
    for jf in json_files:
        print(f"[ingest] Processing {jf.name} ...")
        with open(jf) as f:
            chunks = json.load(f)
        added = ingest_chunks(chunks, collection)
        print(f"  [ingest] Added {added} new chunks from {jf.name}")
        total_added += added

    print(f"\n[ingest] Done. Total chunks added: {total_added}")
    print(f"[ingest] Collection now has {collection.count()} total chunks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed and store CS2 round chunks.")
    ap.add_argument("--file", default=None, help="Specific JSON file to ingest")
    args = ap.parse_args()
    main(file=args.file)

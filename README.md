# CS2 RAG Performance Coach

Analyse your CS2 match demos with a fully local, zero-cost RAG pipeline.
Ask natural language questions and get actionable coaching insights backed
by your actual match data.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.11+ | https://python.org |
| Ollama | https://ollama.com |

---

## 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 2 — Pull Ollama models

```bash
# Embedding model (small, fast)
ollama pull nomic-embed-text

# LLM — pick ONE based on your RAM:
ollama pull llama3.2          # 8 GB RAM  (3B, fastest)
ollama pull mistral           # 16 GB RAM (7B, better reasoning)
ollama pull llama3.1:8b       # 16 GB RAM (8B, good balance)
```

---

## 3 — Download demos from Leetify

```bash
# Download 10 most recent demos (prompts for your Steam64 ID)
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10

# With a Leetify API key (higher rate limits) — get one at leetify.com/app/developer
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10 --api-key YOUR_KEY

# Or set the key as an env var so you don't have to type it each time
export LEETIFY_API_KEY=your_key_here
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10

# Force re-download demos already present
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10 --redownload
```

Demos are saved to `data/demos/` as `<game_id>.dem`, decompressed automatically.

---

## 4 — Parse a demo

Drop your `.dem` files into `data/demos/`, then run:

```bash
python parser/parse_demo.py data/demos/your_match.dem
```

Parsed round chunks land in `data/parsed/your_match.json`.

---

## 4 — Ingest into ChromaDB

```bash
# Ingest all parsed files
python ingest/embed_and_store.py

# Or a specific file
python ingest/embed_and_store.py --file data/parsed/your_match.json
```

---

## 5 — Ask questions

**CLI:**
```bash
python retrieval/query.py "Why do I keep losing pistol rounds?"

# Filter to a specific map
python retrieval/query.py "How is my B site defence?" --map de_mirage
```

**Web UI:**
```bash
python ui/app.py
# Open http://localhost:7860
```

---

## Project Layout

```
cs2-rag/
├── fetcher/
│   └── fetch_demos.py       # Leetify API → download .dem files
├── parser/
│   └── parse_demo.py        # awpy demo → round chunks (JSON)
├── ingest/
│   └── embed_and_store.py   # embed chunks → ChromaDB
├── retrieval/
│   └── query.py             # similarity search + LLM generation
├── ui/
│   └── app.py               # Gradio web interface
├── data/
│   ├── demos/               # place .dem files here
│   ├── parsed/              # auto-generated JSON chunks
│   └── chroma_db/           # auto-generated vector DB
└── requirements.txt
```

---

## Example Questions

- *"Why am I losing pistol rounds?"*
- *"Which map is my worst?"*
- *"Am I making good economic decisions?"*
- *"What weapons kill me most?"*
- *"How often do I win after planting the bomb?"*
- *"When should I be saving instead of force-buying?"*

---

## Tuning Tips

- **More context**: Increase `TOP_K` in `query.py` if answers feel too narrow.
- **Better LLM**: Switch `LLM_MODEL` to `mistral` or `llama3.1:8b` for deeper analysis.
- **Map filters**: Use `--map de_inferno` to focus retrieval on a single map.
- **Re-ingest is safe**: `embed_and_store.py` uses upsert, so re-running won't duplicate.

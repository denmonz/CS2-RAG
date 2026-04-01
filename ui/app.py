"""
app.py
------
Gradio web interface for the CS2 RAG coach.

Run with:
    python ui/app.py

Then open http://localhost:7860 in your browser.
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import chromadb
from chromadb.config import Settings

from retrieval.query import ask, retrieve, CHROMA_PATH, COLLECTION, TOP_K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_available_maps() -> list[str]:
    """Pull distinct map names from ChromaDB metadata."""
    try:
        client     = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION)
        all_meta   = collection.get(include=["metadatas"])["metadatas"]
        maps       = sorted({m.get("map", "unknown") for m in all_meta})
        return ["All maps"] + maps
    except Exception:
        return ["All maps"]


def get_db_stats() -> str:
    try:
        client     = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION)
        count      = collection.count()
        return f"✅ Database ready — {count} round chunks indexed"
    except Exception:
        return "⚠️ Database not found. Run embed_and_store.py first."


def format_retrieved_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No chunks retrieved."
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        lines.append(
            f"**Round {m.get('round','?')}** | {m.get('map','?')} | "
            f"Winner: `{m.get('winner_side','?')}` | "
            f"Kills: {m.get('kill_count','?')} | "
            f"Relevance: {1 - c['distance']:.2%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handle_query(question: str, map_filter: str, top_k: int,
                 show_chunks: bool) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    filters    = None if map_filter == "All maps" else {"map": map_filter}
    chunks     = retrieve(question, top_k=int(top_k), filters=filters)
    chunk_info = format_retrieved_chunks(chunks) if show_chunks else ""

    answer     = ask(question, filters=filters, top_k=int(top_k))
    return answer, chunk_info


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Why am I losing pistol rounds?",
    "Which map is my worst performing?",
    "Am I making good economic decisions?",
    "What weapons am I dying to most often?",
    "How often do I win rounds after planting the bomb?",
    "When do I tend to force-buy when I should save?",
]

with gr.Blocks(
    title="CS2 RAG Coach",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css="""
        .header { text-align: center; padding: 1rem 0; }
        .stats-box { font-size: 0.85rem; color: #aaa; }
    """,
) as demo:

    gr.Markdown("# 🎯 CS2 RAG Performance Coach", elem_classes="header")
    gr.Markdown(
        "Ask natural language questions about your match history. "
        "The coach retrieves relevant rounds from your demos and generates targeted feedback.",
        elem_classes="header",
    )

    db_status = gr.Markdown(get_db_stats, elem_classes="stats-box")

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Why do I keep losing on B site?",
                lines=2,
            )
            with gr.Row():
                map_filter = gr.Dropdown(
                    choices=get_available_maps(),
                    value="All maps",
                    label="Filter by Map",
                    scale=2,
                )
                top_k_slider = gr.Slider(
                    minimum=3, maximum=20, value=TOP_K, step=1,
                    label="Rounds to retrieve (top-k)",
                    scale=3,
                )
            show_chunks = gr.Checkbox(label="Show retrieved rounds", value=True)
            submit_btn  = gr.Button("🔍 Analyse", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("**Example questions:**")
            for eq in EXAMPLE_QUESTIONS:
                gr.Button(eq, size="sm").click(
                    fn=lambda q=eq: q,
                    outputs=question_box,
                )

    with gr.Row():
        answer_box = gr.Markdown(label="Coach Analysis")

    with gr.Row():
        chunks_box = gr.Markdown(label="Retrieved Rounds")

    submit_btn.click(
        fn=handle_query,
        inputs=[question_box, map_filter, top_k_slider, show_chunks],
        outputs=[answer_box, chunks_box],
    )

    question_box.submit(
        fn=handle_query,
        inputs=[question_box, map_filter, top_k_slider, show_chunks],
        outputs=[answer_box, chunks_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

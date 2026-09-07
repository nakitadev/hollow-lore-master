"""Hugging Face Spaces entry point.

HF Spaces only runs ``pip install -r requirements.txt`` — it does NOT run
``pip install -e .``. So the ``lore_master`` package under ``source/`` isn't
importable by default. This file fixes that by adding ``source/`` to ``sys.path``
before importing, then launches the same Gradio UI as ``scripts/app_rag.py``.
"""

import subprocess
import sys
import os
from pathlib import Path

# ให้ Python หา package lore_master ใน source/ เจอ โดยไม่ต้องติดตั้งแบบ editable
ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "source"
sys.path.insert(0, str(SOURCE_DIR))

import gradio as gr

from lore_master.core.components import ensure_pinecone_index
from lore_master.core.config import get_settings
from lore_master.rag_chat.rag_chain import build_rag_chain

# เช็คว่า Pinecone index มีข้อมูลหรือยัง ถ้ายังไม่มีให้ fetch + ingest ใหม่
settings = get_settings()
pinecone_client = ensure_pinecone_index()
index_stats = pinecone_client.Index(settings.pinecone_index_name).describe_index_stats()
if index_stats["total_vector_count"] == 0:
    print("Pinecone index is empty - building vector DB...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "run_fetch_ingest.py")],
        check=True,
        env=env,
    )
chain = build_rag_chain()

theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])


def chat(message: str, history: list[dict], request: gr.Request | None = None):
    thread_id = request.session_hash if (request and getattr(request, "session_hash", None)) else "default"
    partial_text = ""
    for token in chain.stream(
        {"question": message, "history": history},
        config={"configurable": {"thread_id": thread_id}},
    ):
        partial_text += token
        yield partial_text


demo = gr.ChatInterface(
    fn=chat, title="Hollow Knight Lore Bot (RAG, LangChain)"
)

if __name__ == "__main__":
    demo.launch(theme=theme, server_name="0.0.0.0", server_port=7860)
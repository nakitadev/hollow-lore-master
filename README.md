---
title: Hollow Lore Master
emoji: 🦋
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Lore Master (LangChain Edition)

A **Hollow Knight lore Q&A bot** built with **LangChain**. It scrapes lore from
the [Hollow Knight Fandom wiki](https://hollowknight.fandom.com), embeds it into a
[Pinecone](https://www.pinecone.io/) vector store, and answers questions in English
through a Gradio chat UI — grounded in the retrieved lore and citing its sources.

The retrieval chain is **history-aware**: follow-up questions like *"where can I
find him?"* are rewritten into standalone queries before retrieval, so the bot
keeps context across a conversation.

## Architecture

![Hollow Lore Master Architecture](archify/hollow-lore-architecture.png)

> **Interactive Diagram**: Open [`archify/hollow-lore-architecture.html`](archify/hollow-lore-architecture.html) in your browser (or view the [Archify JSON specification](archify/hollow-lore.architecture.json)) for interactive inspection, color modes, and guided views.

| Concern        | Component                                                                 |
|----------------|---------------------------------------------------------------------------|
| Chat model     | `ChatOpenRouter` — `nvidia/nemotron-3.5-lightning:free` ([`core/components.py`](source/lore_master/core/components.py)) |
| Embeddings     | `HuggingFaceEmbeddings` — `all-MiniLM-L6-v2`, local & free                 |
| Fetch lore     | [`rag_chat/fetch_wiki.py`](source/lore_master/rag_chat/fetch_wiki.py) — MediaWiki API → clean `.md` |
| Ingestion      | [`rag_chat/ingest.py`](source/lore_master/rag_chat/ingest.py) — load → split → embed → Pinecone |
| Retrieval      | `build_retriever()` — `vectorstore.as_retriever(k=4)`                      |
| RAG chain      | [`rag_chat/rag_chain.py`](source/lore_master/rag_chat/rag_chain.py) — history-aware LCEL chain |
| UI             | [`scripts/app_rag.py`](scripts/app_rag.py) — `gr.ChatInterface`            |

> `test/test_retriever.py` uses a local, throwaway Chroma store instead of Pinecone
> so the retrieval-logic test stays free and offline — it isn't testing Pinecone
> itself, just that chunking + embedding + retrieval works.

## Setup

Requires Python 3.12+.

```bash
uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .
```

Copy `.env.example` to `.env` and set your OpenRouter and Pinecone keys:

```
OPENROUTER_API_KEY=your-key-here
PINECONE_API_KEY=your-key-here
```

Get a free Pinecone API key at [app.pinecone.io](https://app.pinecone.io/) — no
need to create an index by hand, `build_retriever()` / `create_embeddings()`
create the index (`pinecone_index_name` in `config.py`) on first use if it
doesn't already exist.

> The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB).
> After that, embeddings run locally and offline; only the vector store itself
> is a network call to Pinecone.

## Build the knowledge base

Fetch lore from the wiki and ingest it into the vector store in one step:

```bash
python scripts/run_fetch_ingest.py
```

This:
1. Crawls the wiki recursively starting from `fetch_wiki.ROOT_CATEGORY` (every
   sub-category found along the way is followed too) and saves clean `.md`
   files into `data/knowledge-base/`, mirroring the category hierarchy as
   nested folders. Pages already saved from a previous run are skipped.
2. Splits them into chunks, embeds them, and (re)upserts them into the Pinecone
   index named by `pinecone_index_name` in `config.py`.

> Re-running clears every vector already in the index first, then re-upserts
> from scratch.

## Run the chatbot

```bash
python scripts/app_rag.py
```

Launches a Gradio web UI that answers Hollow Knight questions grounded in the
ingested lore, cites source files, and declines to guess when the answer isn't in
context.

## Configuration

All settings live in [`source/lore_master/core/config.py`](source/lore_master/core/config.py):
model, `temperature`, number of chunks retrieved (`retrieval_k`), the Pinecone
index name/cloud/region, wiki categories, and the knowledge-base path.

## Deploy to Hugging Face Spaces

The YAML header at the top of this file configures a **Gradio Space** whose entry
point is [`app.py`](app.py). To deploy:

1. Create a new **Gradio** Space and push this repo to it.
2. In the Space **Settings → Secrets**, add `OPENROUTER_API_KEY` and `PINECONE_API_KEY`.
3. Commit `data/knowledge-base/` so the Space has the scraped wiki pages without
   running the fetch step. `app.py` checks the Pinecone index on startup and
   automatically runs `scripts/run_fetch_ingest.py` to populate it if it's empty
   (first boot only — later restarts see existing vectors and skip straight to
   launching the chat UI).

> `app.py` adds `source/` to `sys.path` itself, so the Space works with a plain
> `pip install -r requirements.txt` — no editable install (`pip install -e .`)
> needed.

## What LangChain abstracts

Compared to a from-scratch build, LangChain collapses a lot of hand-written
plumbing into framework calls: the retry/backoff loop becomes
`max_retries=3` on the chat model; manual chunking becomes
`RecursiveCharacterTextSplitter`; upserting into Pinecone and building
ids/metadata becomes `PineconeVectorStore.from_documents(...)`; and the whole
retrieve → rewrite → prompt → answer flow becomes a single LCEL pipe with
`RunnablePassthrough.assign`. Less code to maintain, but more behavior hidden
behind abstractions.

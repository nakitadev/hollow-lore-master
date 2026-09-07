import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model: str = "nvidia/nemotron-3.5-lightning:free" 
    max_tokens: int = 1024
    temperature: float = 0.2
    retrieval_k: int = 4                         # number of chunks to retrieve
    chunk_size: int = 800
    chunk_overlap: int = 150
    knowledge_dir: str = "data/knowledge-base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384               # must match embedding_model's output size
    pinecone_index_name: str = "hollow-knight-lore"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"           # Pinecone's free Starter plan only allows serverless indexes here


def get_settings() -> Settings:
    return Settings()


def require_openrouter_key() -> None:
    """Only the chat model needs this - embeddings/fetch/ingest run without it."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not found. Copy .env.example to .env.")


def require_pinecone_key() -> None:
    """Only the vector store needs this - embeddings run locally without it."""
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY not found. Copy .env.example to .env.")

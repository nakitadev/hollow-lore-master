import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from .config import get_settings, require_openrouter_key, require_pinecone_key


def build_chat_model() -> ChatOpenAI:
    s = get_settings()
    require_openrouter_key()
    return ChatOpenAI(
        model=s.model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        max_tokens=s.max_tokens,
        temperature=s.temperature,
        max_retries=3,
        request_timeout=120,
        extra_body={"reasoning": {"max_tokens": 0}},
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    return HuggingFaceEmbeddings(model_name=s.embedding_model)


def ensure_pinecone_index() -> Pinecone:
    """Return a Pinecone client, creating the index on first use if needed."""
    s = get_settings()
    require_pinecone_key()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = {index.name for index in pc.list_indexes()}
    if s.pinecone_index_name not in existing:
        pc.create_index(
            name=s.pinecone_index_name,
            dimension=s.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
    return pc


def build_retriever():
    s = get_settings()
    ensure_pinecone_index()
    vectorstore = PineconeVectorStore(
        index_name=s.pinecone_index_name,
        embedding=build_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": s.retrieval_k})

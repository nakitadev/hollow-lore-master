from typing import Any
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from lore_master.core.components import build_chat_model, build_retriever

RAG_SYSTEM_PROMPT = (
    """You answer questions about Hollow Knight lore using ONLY the provided
context and conversation history. Answer in English. If the answer is not in
the context, say you don't know rather than guessing. Cite the sources you
used by their filename in brackets (e.g. [source: filename.md]).

If the user is merely greeting you, saying hello, or engaging in casual
conversation (e.g., "hi", "hello", "who are you"), respond warmly and naturally
as the Hollow Knight Lore Master and invite them to ask about Hollow Knight lore.
In such cases, do NOT cite sources or bring up the retrieved context.

IMPORTANT: Do NOT output any internal thoughts, reasoning steps, analysis, or preambles (such as "Here's a thinking process:"). Output ONLY your direct, final response."""
)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


class RAGState(MessagesState):
    """LangGraph state keeping short-term messages and retrieved RAG context."""
    context: str


class RAGGraphWrapper:
    """Convenience wrapper exposing an LCEL-compatible .invoke() and .stream() interface."""

    def __init__(self, graph):
        self.graph = graph

    def invoke(self, inputs: dict[str, Any] | str, config: dict[str, Any] | None = None) -> Any:
        thread_id = (
            config.get("configurable", {}).get("thread_id", "default")
            if config
            else "default"
        )
        cfg = {"configurable": {"thread_id": thread_id}}

        if isinstance(inputs, str):
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=inputs)]},
                cfg,
            )
            return result["messages"][-1].content
        elif isinstance(inputs, dict):
            if "question" in inputs and "messages" not in inputs:
                result = self.graph.invoke(
                    {"messages": [HumanMessage(content=inputs["question"])]},
                    cfg,
                )
                return result["messages"][-1].content
            return self.graph.invoke(inputs, cfg)
        return self.graph.invoke(inputs, cfg)

    def stream(self, inputs: dict[str, Any] | str, config: dict[str, Any] | None = None):
        """Stream response tokens as a synchronous generator."""
        thread_id = (
            config.get("configurable", {}).get("thread_id", "default")
            if config
            else "default"
        )
        cfg = {"configurable": {"thread_id": thread_id}}

        if isinstance(inputs, str):
            graph_inputs = {"messages": [HumanMessage(content=inputs)]}
        elif isinstance(inputs, dict):
            if "messages" in inputs:
                graph_inputs = inputs
            elif "question" in inputs:
                graph_inputs = {"messages": [HumanMessage(content=inputs["question"])]}
            else:
                graph_inputs = inputs
        else:
            graph_inputs = inputs

        for chunk, metadata in self.graph.stream(
            graph_inputs,
            cfg,
            stream_mode="messages",
        ):
            if (
                metadata.get("langgraph_node") == "generate"
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
            ):
                yield chunk.content


def build_rag_chain() -> RAGGraphWrapper:
    """Build a StateGraph RAG pipeline using InMemorySaver for short-term memory."""
    retriever = build_retriever()
    model = build_chat_model()

    def retrieve(state: RAGState) -> dict:
        user_query = state["messages"][-1].content
        docs = retriever.invoke(user_query)
        return {"context": format_docs(docs)}

    def generate(state: RAGState) -> dict:
        context = state.get("context", "")
        system_prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"
        # Inject retrieved context into system prompt alongside the short-term conversation memory
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        chunks = []
        for chunk in model.stream(messages):
            chunks.append(chunk)
        if not chunks:
            return {"messages": [AIMessage(content="")]}
        full_message = sum(chunks[1:], chunks[0])
        return {"messages": [AIMessage(content=full_message.content.strip())]}

    # Construct the state graph
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # Short-term memory checkpointer: stores state per thread_id
    checkpointer = InMemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)

    return RAGGraphWrapper(compiled_graph)

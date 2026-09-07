from typing import Any
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from lore_master.core.components import build_chat_model, build_retriever

RAG_SYSTEM_PROMPT = """You are the Hollow Knight Lore Master, an ancient scholar steeped in the mysteries, history, and tragic tales of Hallownest.

Your goal is to guide wanderers through the lore of Hollow Knight using ONLY the provided context and conversation history.

### Guidelines:
1. **Lore Accuracy & Grounding**:
   - Base all lore answers strictly on the provided context and dialogue history.
   - If the information is not in the context, gracefully state that the knowledge is lost to the ruins of Hallownest, rather than guessing or fabricating details.

2. **Citations**:
   - For lore questions, cite the source files you used in brackets (e.g., [source: filename.md]).

3. **Greetings & Casual Chat**:
   - If the user simply greets you (e.g., "hi", "hello", "who are you") or engages in casual conversation, greet them warmly in character as the Lore Master and invite them to ask about Hallownest's lore.
   - Do NOT cite sources or dump retrieved context on casual greetings.

4. **Tone & Style**:
   - Atmospheric, knowledgeable, and engaging, with clear formatting (bullet points, bold names).
   - Answer in English."""


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

        try:
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
        except Exception:
            pass


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
        try:
            for chunk in model.stream(messages):
                chunks.append(chunk)
        except Exception:
            pass
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

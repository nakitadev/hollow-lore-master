import gradio as gr

from lore_master.rag_chat.rag_chain import build_rag_chain

chain = build_rag_chain()

theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])


def chat(message: str, history: list[dict], request: gr.Request | None = None) -> str:
    thread_id = request.session_hash if (request and getattr(request, "session_hash", None)) else "default"
    return chain.invoke(
        {"question": message, "history": history},
        config={"configurable": {"thread_id": thread_id}},
    )


demo = gr.ChatInterface(
    fn=chat, title="Hollow Knight Lore Bot (RAG, LangChain)"
)

if __name__ == "__main__":
    demo.launch(theme=theme)

import os
import sys
import uuid
from typing import List, Tuple, Optional

import gradio as gr

# LangSmith tracing setup via environment variables
# Required env vars: LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
# We enable tracing only if both are set
if os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ---- Imports from LangChain ecosystem ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import InMemoryChatMessageHistory

# LLMs: Ollama or HuggingFaceEndpoint
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint


# ------------------ Configuration ------------------
PDF_PATH = os.environ.get("HISTORYBOT_PDF", "/workspace/historical_figures.pdf")

# Vector store configuration
VECTORSTORE_TYPE = os.environ.get("HISTORYBOT_VECTORSTORE", "chroma").lower()
CHROMA_PERSIST_DIR = os.environ.get("HISTORYBOT_CHROMA_DIR", "/workspace/chroma_history")
CHROMA_COLLECTION_NAME = os.environ.get("HISTORYBOT_COLLECTION", "historical_figures")

# Embeddings (Ollama)
OLLAMA_EMBED_MODEL = os.environ.get("HISTORYBOT_EMBED_MODEL", "granite-embedding:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# LLM provider selection
LLM_PROVIDER = os.environ.get("HISTORYBOT_LLM_PROVIDER", "ollama").lower()  # options: ollama|huggingface

# Ollama model
OLLAMA_LLM_MODEL = os.environ.get("HISTORYBOT_OLLAMA_MODEL", "llama3:latest")

# HuggingFaceEndpoint
HF_REPO_ID = os.environ.get("HISTORYBOT_HF_REPO_ID", "Menlo/Jan-nano-128k")
HF_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Text splitting
CHUNK_SIZE = int(os.environ.get("HISTORYBOT_CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.environ.get("HISTORYBOT_CHUNK_OVERLAP", "30"))

# Retriever
NUM_RETRIEVAL_DOCS = int(os.environ.get("HISTORYBOT_TOP_K", "4"))

# UI
GREETING = (
    "Hello, I am HistoryBot, your expert on historical figures. How can I assist you today?"
)


# ------------------ Utilities ------------------
def ensure_pdf_exists(pdf_path: str) -> bool:
    if not os.path.isfile(pdf_path):
        sys.stderr.write(
            f"[HistoryBot] PDF not found at {pdf_path}. Please place 'historical_figures.pdf' at that path or set HISTORYBOT_PDF.\n"
        )
        return False
    return True


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    documents = splitter.split_documents(pages)
    return documents


def init_embeddings() -> OllamaEmbeddings:
    # Requires a running Ollama server with the embedding model pulled
    # e.g., `ollama pull granite-embedding:latest`
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return embeddings


def init_vectorstore(documents: List[Document], embeddings: OllamaEmbeddings) -> Chroma:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.persist()
    return vectorstore


def init_llm():
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    elif LLM_PROVIDER == "huggingface":
        if not HF_API_TOKEN:
            raise RuntimeError(
                "HUGGINGFACEHUB_API_TOKEN not set. Please set it to use HuggingFaceEndpoint."
            )
        return HuggingFaceEndpoint(
            repo_id=HF_REPO_ID,
            temperature=0.2,
            max_new_tokens=512,
        )
    else:
        raise ValueError("Unsupported LLM provider. Use 'ollama' or 'huggingface'.")


def build_prompt() -> PromptTemplate:
    template = (
        "You are HistoryBot, an expert assistant about historical figures.\n"
        "Use the provided context to answer the user's question.\n"
        "If the answer cannot be found in the context, say you do not know.\n"
        "Keep answers concise and factual.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def format_chat_history(history: InMemoryChatMessageHistory, max_turns: int = 6) -> str:
    # Return the most recent N turns formatted as plain text
    if not history or not history.messages:
        return ""
    turns = []
    # messages are alternating HumanMessage / AIMessage ideally
    for msg in history.messages[-(max_turns * 2) :]:
        role = getattr(msg, "type", None) or msg.__class__.__name__.replace("Message", "")
        role = "User" if role.lower() in ("human", "aih","humanmessage") else ("Assistant" if role.lower() in ("ai", "aimessage") else role)
        turns.append(f"{role}: {msg.content}")
    return "\n".join(turns)


def build_qa_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVAL_DOCS})
    llm = init_llm()
    prompt = build_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
        input_key="query",
        output_key="result",
    )
    return qa_chain


# ------------------ Global (lazy) state ------------------
_vectorstore: Optional[Chroma] = None
_qa_chain = None


def initialize_pipeline_if_needed():
    global _vectorstore, _qa_chain
    if _qa_chain is not None and _vectorstore is not None:
        return

    if not ensure_pdf_exists(PDF_PATH):
        # Delay hard failure to first query, allowing UI to load
        return

    documents = load_and_split_pdf(PDF_PATH)
    embeddings = init_embeddings()
    _vectorstore = init_vectorstore(documents, embeddings)
    _qa_chain = build_qa_chain(_vectorstore)


# ------------------ Gradio Handlers ------------------

def answer_question(user_text: str, chat_ui_history: List[Tuple[str, str]], history_obj: Optional[InMemoryChatMessageHistory]):
    # Ensure pipeline is initialized (lazy)
    try:
        initialize_pipeline_if_needed()
    except Exception as e:
        err = (
            "Initialization error. Ensure Ollama is running with the embedding and LLM models pulled, "
            "the PDF exists, and environment variables are configured. Error: " + str(e)
        )
        chat_ui_history = chat_ui_history + [(user_text, err)]
        return chat_ui_history, history_obj

    if _qa_chain is None:
        # Likely PDF missing
        msg = (
            "Knowledge base not initialized. Please ensure 'historical_figures.pdf' exists at "
            f"{PDF_PATH} and that Ollama is running (embedding model '{OLLAMA_EMBED_MODEL}' pulled)."
        )
        chat_ui_history = chat_ui_history + [(user_text, msg)]
        return chat_ui_history, history_obj

    if history_obj is None:
        history_obj = InMemoryChatMessageHistory()

    # Add the user message to memory first
    history_obj.add_user_message(user_text)

    # Inject formatted conversation history into the question so RetrievalQA can use it via the prompt
    history_text = format_chat_history(history_obj)
    composed_query = (
        ("Conversation so far:\n" + history_text + "\n\n" if history_text else "")
        + "User question: "
        + user_text
    )

    try:
        result = _qa_chain({"query": composed_query})
        answer = result.get("result", "I was unable to generate an answer.")
    except Exception as e:
        answer = f"An error occurred during retrieval or generation: {e}"

    # Add assistant reply to memory
    history_obj.add_ai_message(answer)

    chat_ui_history = chat_ui_history + [(user_text, answer)]
    return chat_ui_history, history_obj


def clear_history():
    return [], InMemoryChatMessageHistory()


# ------------------ App ------------------

def build_ui():
    with gr.Blocks(title="HistoryBot") as demo:
        gr.Markdown(f"**{GREETING}**")

        chatbot = gr.Chatbot(label="HistoryBot", height=400, type="tuples")
        user_input = gr.Textbox(label="Your question", placeholder="Ask about a historical figure...")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear History")

        history_state = gr.State()  # holds InMemoryChatMessageHistory

        submit_btn.click(
            fn=answer_question,
            inputs=[user_input, chatbot, history_state],
            outputs=[chatbot, history_state],
        )
        user_input.submit(
            fn=answer_question,
            inputs=[user_input, chatbot, history_state],
            outputs=[chatbot, history_state],
        )

        clear_btn.click(fn=clear_history, inputs=None, outputs=[chatbot, history_state])

    return demo


def main():
    # Trigger lazy init in background so first request is faster (non-fatal if it fails)
    try:
        initialize_pipeline_if_needed()
    except Exception as e:
        sys.stderr.write(f"[HistoryBot] Lazy init warning: {e}\n")

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
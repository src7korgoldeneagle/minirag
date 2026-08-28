import os
from pathlib import Path
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from ingest import ingest_documents

# ============================================================
# 1. Page setup
# ============================================================
st.set_page_config(page_title="Mini RAG", page_icon="📚")
st.title("📚 Mini RAG")

# ============================================================
# 2. Load API key (Streamlit secrets first, env var as fallback)
# ============================================================
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
requested_model = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL"))
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit's Secrets settings.")
    st.stop()

# ============================================================
# 3. Cache the heavy resources so they load once, not every rerun
# ============================================================
@st.cache_resource
def load_google_client():
    return genai.Client(api_key=api_key)

@st.cache_resource
def load_generation_model():
    if requested_model:
        return requested_model

    available_models = google_client.models.list()
    preferred_models = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    )
    available_names = {
        model.name.removeprefix("models/")
        for model in available_models
        if "generateContent" in (model.supported_actions or [])
    }

    for model_name in preferred_models:
        if model_name in available_names:
            return model_name

    raise RuntimeError(
        "No Gemini model available for generateContent. "
        "Set GEMINI_MODEL to a model enabled for your Google API key."
    )

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    database_path = Path(__file__).resolve().parent / "chroma_db"
    documents_path = Path(__file__).resolve().parent / "documents"
    chroma_client = chromadb.PersistentClient(path=str(database_path))
    collection = chroma_client.get_or_create_collection(name="documents")

    if collection.count() == 0:
        ingest_documents(
            folder=str(documents_path),
            embedding_model=embedding_model,
            collection=collection
        )

    if collection.count() == 0:
        raise RuntimeError(
            f"No indexed documents found in {documents_path}. "
            "Add a PDF to documents/ and run ingest.py."
        )

    return collection

google_client = load_google_client()
generation_model = load_generation_model()
embedding_model = load_embedding_model()
collection = load_collection()

# ============================================================
# 4. Retrieve relevant chunks (unchanged)
# ============================================================
def retrieve_documents(question):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return documents, metadatas

# ============================================================
# 5. Generate answer (unchanged)
# ============================================================
def generate_answer(question, documents, metadatas):
    context_parts = []
    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"""
Source: {metadata['source']}
Chunk: {metadata['chunk']}
{document}
"""
        )
    context = "\n".join(context_parts)
    prompt = f"""
You are a document question-answering assistant.
Answer the question using ONLY the provided context.
Do not use outside knowledge.
Do not make up information.
If the answer cannot be found in the context,
say:
"I couldn't find the answer in the provided documents."
Keep the answer clear and concise.

==============================
CONTEXT
==============================
{context}

==============================
QUESTION
==============================
{question}
"""
    response = google_client.models.generate_content(
        model=generation_model,
        contents=prompt
    )
    return response.text or "I couldn't generate an answer."

# ============================================================
# 6. Chat UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} (chunk {s['chunk']})")
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            documents, metadatas = retrieve_documents(question)
            answer = generate_answer(question, documents, metadatas)
        st.write(answer)
        with st.expander("Sources"):
            for m in metadatas:
                st.write(f"- {m['source']} (chunk {m['chunk']})")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": metadatas
    })
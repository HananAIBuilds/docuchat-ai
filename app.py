"""
app.py
------
DocuChat AI - chat with your documents using Retrieval-Augmented Generation.

Streamlit front-end wired to rag_engine.RAGEngine. Handles file upload,
chunking + embedding (cached per file so re-asking questions is instant),
a chat-style Q&A interface, and transparent "sources used" for every answer.

Note: embeddings run locally via sentence-transformers (no API key needed).
Only a Google API key is required, for answer generation.
"""

import os
import streamlit as st

from file_loader import extract_text, get_extension, SUPPORTED_EXTENSIONS
from rag_engine import RAGEngine, compute_file_hash

# --------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="DocuChat AI - RAG File Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_CHUNKS_WARNING = 300  # warn if a document will require this many embedding calls


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def get_key(name: str) -> str | None:
    """Look for a key in Streamlit secrets first (used on Streamlit Cloud),
    then fall back to an environment variable (used locally)."""
    key = None
    try:
        key = st.secrets.get(name)
    except Exception:
        key = None
    if not key:
        key = os.environ.get(name)
    return key


def init_session_state():
    defaults = {
        "engine": None,
        "file_hash": None,
        "file_name": None,
        "num_chunks": 0,
        "messages": [],  # [{role, content, sources?}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation():
    st.session_state.messages = []


def reset_document():
    st.session_state.engine = None
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.num_chunks = 0
    st.session_state.messages = []


init_session_state()

# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📄 DocuChat AI")
    st.caption("Retrieval-Augmented Generation over your own documents")

    google_api_key = get_key("GOOGLE_API_KEY")

    if google_api_key:
        st.success("Google API key loaded", icon="✅")
    else:
        st.error("Missing GOOGLE_API_KEY", icon="⚠️")
        st.caption(
            "Add it to `.streamlit/secrets.toml` locally, or under "
            "**App settings → Secrets** on Streamlit Cloud."
        )

    st.divider()

    st.markdown("### ⚙️ Settings")
    top_k = st.slider(
        "Chunks retrieved per question (k)",
        min_value=1,
        max_value=10,
        value=5,
        help="How many of the most relevant chunks are sent to the model as context.",
    )

    st.divider()

    with st.expander("ℹ️ How this works"):
        st.markdown(
            """
1. **Chunking** - your document is split into overlapping text windows.
2. **Embedding** - each chunk is converted into a vector **locally** (`all-MiniLM-L6-v2`, no external API call).
3. **Indexing** - vectors are stored in a FAISS similarity index.
4. **Retrieval** - your question is embedded and matched against the top-k
   closest chunks.
5. **Generation** - `gemini-2.5-flash` answers using only those chunks as context.
            """
        )

    with st.expander("⚠️ Limitations"):
        st.markdown(
            """
- This tool retrieves the **most relevant chunks**, not the entire file,
  it's built for *"look up this specific fact"* questions, not
  *"count/aggregate across every row"* questions.
- Very large files are processed chunk-by-chunk, so processing time scales
  with document size.
- Answers are only as accurate as the source document and the retrieved chunks.
- Embeddings run locally on-device specifically to avoid depending on an
  external embedding API mid-conversation, see the README for why.
            """
        )

    st.divider()
    st.caption("Built with Streamlit · FAISS · sentence-transformers · Gemini")
    st.caption("[View source on GitHub](https://github.com/HananAIBuilds/docuchat-ai/tree/main)")

# --------------------------------------------------------------------- #
# Main header
# --------------------------------------------------------------------- #
st.title("📄 DocuChat AI")
st.markdown(
    "Upload a document and ask questions about it, answers are grounded "
    "strictly in your file's content, with sources shown for every response."
)

if not google_api_key:
    st.stop()

# --------------------------------------------------------------------- #
# File upload + processing
# --------------------------------------------------------------------- #
uploaded_file = st.file_uploader(
    "Upload a document",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)

    # Only (re)process if this is a genuinely new/different file
    if file_hash != st.session_state.file_hash:
        try:
            with st.spinner("Reading document..."):
                raw_text = extract_text(uploaded_file.name, file_bytes)

            if not raw_text or not raw_text.strip():
                st.error("No extractable text was found in this file.")
                st.stop()

            ext = get_extension(uploaded_file.name)
            chunk_size, overlap = RAGEngine.get_chunk_params(ext)
            chunks = RAGEngine.chunk_text(raw_text, chunk_size, overlap)

            if len(chunks) > MAX_CHUNKS_WARNING:
                st.warning(
                    f"This document will generate **{len(chunks)} chunks** "
                    f"(one embedding call each). This may take a while, consider "
                    f"trimming the file if this is unexpected."
                )

            progress_bar = st.progress(0.0, text="Generating embeddings...")

            def _progress(frac):
                progress_bar.progress(frac, text=f"Generating embeddings... {int(frac * 100)}%")

            engine = RAGEngine(google_api_key)
            engine.build_index(chunks, progress_callback=_progress)
            progress_bar.empty()

            st.session_state.engine = engine
            st.session_state.file_hash = file_hash
            st.session_state.file_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)
            st.session_state.messages = []

            st.success(
                f"**{uploaded_file.name}** processed into "
                f"**{len(chunks)} chunks** and ready for questions."
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:  # noqa: BLE001
            st.error(f"Something went wrong while processing this file: {e}")
            st.stop()
    else:
        st.info(
            f"**{st.session_state.file_name}** is already loaded "
            f"({st.session_state.num_chunks} chunks) - ask away below.",
            icon="✅",
        )

# --------------------------------------------------------------------- #
# Chat interface
# --------------------------------------------------------------------- #
if st.session_state.engine is not None:
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear chat", use_container_width=True):
            reset_conversation()
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources used for this answer"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Chunk {i}** · distance `{src['distance']:.3f}`")
                        st.text(
                            src["text"][:500] + ("..." if len(src["text"]) > 500 else "")
                        )
                        st.divider()

    if user_query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.engine.generate_answer(
                        user_query, k=top_k
                    )
                except Exception as e:  # noqa: BLE001
                    answer, sources = f"⚠️ Error generating an answer: {e}", []

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources used for this answer"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"**Chunk {i}** · distance `{src['distance']:.3f}`")
                        st.text(
                            src["text"][:500] + ("..." if len(src["text"]) > 500 else "")
                        )
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
else:
    st.info("👆 Upload a document above to start asking questions.")

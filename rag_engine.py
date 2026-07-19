"""
rag_engine.py
-------------
Core Retrieval-Augmented Generation engine.

Pipeline:
1. Chunking   -> split raw text into overlapping windows
2. Embedding  -> local sentence-transformer, with retry-on-failure
3. Indexing   -> FAISS flat L2 index (in-memory vector store)
4. Retrieval  -> top-k nearest chunks for a query
5. Generation -> Gemini generation model, grounded on retrieved chunks,
                 with automatic fallback to Groq if Gemini fails

This module has zero Streamlit imports on purpose, it can be reused in a
CLI, a notebook, or a different UI framework without modification.

Embedding provider note (read before touching the embedding step):

This went through two embedding providers before landing here. It started
on Gemini's own embedding model, sharing one Google API key/quota with
generation, every chunk of every uploaded document needs its own embedding
call, so that quota got exhausted fast, and once it did, the whole app
stalled (no embeddings -> nothing to search -> no answers).

The first fix moved embeddings to HuggingFace's hosted Inference API
(`HuggingFaceEndpointEmbeddings`), which solved the shared-quota problem but
introduced a new one: that hosted endpoint intermittently returns `504`
(server busy/timeout) errors under load. Not constant, but frequent enough
that a live deployment could hit it at any moment, with a real visitor
watching it fail mid-request. That's a fine risk for a local script (just
re-run it) but not for a deployed app.

Embeddings now run locally via `sentence-transformers`
(`HuggingFaceEmbeddings`) instead, the model downloads once into the app's
environment, and every embedding call after that runs on-device with no
external API call and nothing to time out, rate-limit, or exhaust a quota
on. No HuggingFace token is needed anymore either. The trade-off is a
slightly longer cold start on first load and a small, fixed memory
footprint (~90MB for all-MiniLM-L6-v2), both fine at this project's scale.

Generation provider note:

Generation stayed on Gemini (`gemini-2.5-flash`) since embeddings, not
generation, were the original bottleneck. But Gemini's free tier has a very
low daily request cap (20/day at time of writing), and generation still
runs on every single chat question, so a normal conversation can exhaust
it within a handful of messages. When that happens, `generate_answer` now
automatically retries the same prompt on Groq (`llama-3.1-8b-instant`)
instead of raising, so a quota hit degrades gracefully into "still works,
answered by a different model" rather than a hard failure or an ugly error
shown to the user mid-conversation. If Groq isn't configured, or also
fails, a friendly message is returned instead of propagating the exception.
"""

import time
import hashlib
import numpy as np
import faiss
from google import genai
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings


class RAGEngine:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GEN_MODEL = "gemini-2.5-flash"
    FALLBACK_GEN_MODEL = "llama-3.1-8b-instant"

    # Tabular data (csv/xlsx) is kept in larger, less-overlapping chunks so that
    # rows / records aren't fragmented mid-way. Prose files use smaller,
    # denser chunks for tighter semantic matches.
    CHUNK_PARAMS = {
        ".csv": (4500, 250),
        ".xlsx": (4500, 250),
        "default": (800, 100),
    }

    def __init__(self, google_api_key: str, groq_api_key: str | None = None):
        if not google_api_key:
            raise ValueError(
                "A Google API key is required to initialize RAGEngine (used for answer generation)."
            )
        self.client = genai.Client(api_key=google_api_key)
        self.groq_api_key = groq_api_key
        self.embedder = HuggingFaceEmbeddings(model_name=self.EMBED_MODEL)
        self.chunks = []
        self.index = None
        self.embeddings = None

    # ------------------------------------------------------------------ #
    #                             Chunking                               
    # ------------------------------------------------------------------ #
    @classmethod
    def get_chunk_params(cls, file_ext: str):
        return cls.CHUNK_PARAMS.get(file_ext, cls.CHUNK_PARAMS["default"])

    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            start += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------ #
    #                                 Embedding
    # ------------------------------------------------------------------ #
    def _embed_single(self, text: str, max_retries: int = 3, retry_wait: float = 8.0):
        """Embeds one chunk/query locally. Retry logic is kept even though
        local inference has nothing to rate-limit, a transient failure
        (e.g. first-load model download hiccup) is still worth retrying."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.embedder.embed_query(text)
            except Exception as e:  # noqa: BLE001 - want to retry on any transient error
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_wait)
        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_error}")

    def build_index(self, chunks: list[str], progress_callback=None):
        """Embed every chunk and build a FAISS flat-L2 index over them.

        progress_callback: optional callable(fraction_complete: float) -> None,
        used by the UI to drive a progress bar.
        """
        if not chunks:
            raise ValueError("No chunks to index, the document may be empty.")

        embeddings = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            embeddings.append(self._embed_single(chunk))
            if progress_callback:
                progress_callback((i + 1) / total)

        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        self.chunks = chunks
        self.embeddings = embeddings_array
        self.index = index
        return index

    # ------------------------------------------------------------------ #
    #                               Retrieval
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("Index has not been built yet, process a document first.")

        query_embedding = self._embed_single(query)
        query_array = np.array([query_embedding]).astype("float32")

        k = max(1, min(k, len(self.chunks)))
        distances, indices = self.index.search(query_array, k)

        return [
            {"text": self.chunks[idx], "distance": float(dist)}
            for idx, dist in zip(indices[0], distances[0])
        ]

    # ------------------------------------------------------------------ #
    #                               Generation
    # ------------------------------------------------------------------ #
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful assistant answering questions about a document.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer confidently, say so honestly instead of guessing.
Do not make up facts, numbers, or figures that are not present in the context.

Context:    {context}

Question:   {query}

Answer:"""

    def _generate_with_fallback(self, prompt: str) -> str:
        """Try Gemini first; if it fails for any reason (quota, network, etc.),
        fall back to Groq. If both fail, return a friendly message instead of
        raising, so the chat UI never has to show a raw exception."""
        try:
            response = self.client.models.generate_content(
                model=self.GEN_MODEL,
                contents=prompt,
            )
            return response.text

        except Exception as e:  # noqa: BLE001 - want to catch any Gemini failure and fall back
            print(f"Gemini generation failed, falling back to Groq. Error: {e}")

            if not self.groq_api_key:
                return (
                    "I apologize, the AI service (Gemini) is temporarily unavailable, "
                    "and no backup is configured. Please try again shortly."
                )

            try:
                groq_client = Groq(api_key=self.groq_api_key)
                groq_response = groq_client.chat.completions.create(
                    model=self.FALLBACK_GEN_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return groq_response.choices[0].message.content

            except Exception as groq_error:  # noqa: BLE001
                print(f"Groq also failed. Error: {groq_error}")
                return "I apologize, our AI service is temporarily unavailable. Please try again in a moment."

    def generate_answer(self, query: str, k: int = 5):
        results = self.search(query, k=k)
        context = "\n\n---\n\n".join(r["text"] for r in results)
        prompt = self._build_prompt(query, context)
        answer_text = self._generate_with_fallback(prompt)
        return answer_text, results


def compute_file_hash(file_bytes: bytes) -> str:
    """Stable fingerprint used to detect whether the uploaded file changed,
    so we don't re-embed (and re-pay for) the same document twice."""
    return hashlib.md5(file_bytes).hexdigest()

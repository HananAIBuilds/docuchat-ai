"""
rag_engine.py
-------------
Core Retrieval-Augmented Generation engine.

Pipeline:
    1. Chunking      -> split raw text into overlapping windows
    2. Embedding     -> HuggingFace sentence-transformer, with retry-on-failure
    3. Indexing      -> FAISS flat L2 index (in-memory vector store)
    4. Retrieval     -> top-k nearest chunks for a query
    5. Generation    -> Gemini generation model, grounded on retrieved chunks

This module has zero Streamlit imports on purpose — it can be reused in a
CLI, a notebook, or a different UI framework without modification.

Embedding provider note (read before touching the embedding step):
This originally used Gemini's own embedding model (`gemini-embedding-001`)
for both embeddings and generation, sharing one Google API key/quota. In
production, embeddings burn through that quota far faster than generation
does — every chunk of every uploaded document needs its own embedding call,
on top of one more call per question asked. Once that quota was exhausted,
the entire app stalled: no embeddings -> nothing to search -> no answers —
even though the generation quota was completely untouched.

Embeddings were moved to a HuggingFace-hosted sentence-transformer instead
(free, separate rate limit, no shared quota with generation), so the two
steps now fail independently rather than one quota outage taking the whole
pipeline down. Generation stays on Gemini (`gemini-2.5-flash`), since that
wasn't the bottleneck.
"""

import time
import hashlib
import numpy as np
import faiss
from google import genai
from langchain_huggingface import HuggingFaceEndpointEmbeddings


class RAGEngine:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GEN_MODEL = "gemini-2.5-flash"

    # Tabular data (csv/xlsx) is kept in larger, less-overlapping chunks so that
    # rows / records aren't fragmented mid-way. Prose files use smaller,
    # denser chunks for tighter semantic matches.
    CHUNK_PARAMS = {
        ".csv": (4500, 250),
        ".xlsx": (4500, 250),
        "default": (800, 100),
    }

    def __init__(self, google_api_key: str, hf_token: str):
        if not google_api_key:
            raise ValueError(
                "A Google API key is required to initialize RAGEngine (used for answer generation)."
            )
        if not hf_token:
            raise ValueError(
                "A HuggingFace token is required to initialize RAGEngine (used for embeddings)."
            )

        self.client = genai.Client(api_key=google_api_key)
        self.embedder = HuggingFaceEndpointEmbeddings(
            model=self.EMBED_MODEL,
            huggingfacehub_api_token=hf_token,
        )
        self.chunks = []
        self.index = None
        self.embeddings = None

    # ------------------------------------------------------------------ #
    # Chunking
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
    # Embedding
    # ------------------------------------------------------------------ #
    def _embed_single(self, text: str, max_retries: int = 3, retry_wait: float = 8.0):
        """Embeds one chunk/query via the HuggingFace endpoint, retrying on
        transient failures (kept from the original Gemini implementation —
        the retry pattern itself didn't need to change, just the client)."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.embedder.embed_query(text)
            except Exception as e:  # noqa: BLE001 - want to retry on any transient API error
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
            raise ValueError("No chunks to index — the document may be empty.")

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
    # Retrieval
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("Index has not been built yet — process a document first.")

        query_embedding = self._embed_single(query)
        query_array = np.array([query_embedding]).astype("float32")

        k = max(1, min(k, len(self.chunks)))
        distances, indices = self.index.search(query_array, k)

        return [
            {"text": self.chunks[idx], "distance": float(dist)}
            for idx, dist in zip(indices[0], distances[0])
        ]

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate_answer(self, query: str, k: int = 5):
        results = self.search(query, k=k)
        context = "\n\n---\n\n".join(r["text"] for r in results)

        prompt = f"""You are a helpful assistant answering questions about a document.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer confidently, say so honestly instead of guessing.
Do not make up facts, numbers, or figures that are not present in the context.

Context:
{context}

Question: {query}

Answer:"""

        response = self.client.models.generate_content(
            model=self.GEN_MODEL,
            contents=prompt,
        )
        return response.text, results


def compute_file_hash(file_bytes: bytes) -> str:
    """Stable fingerprint used to detect whether the uploaded file changed,
    so we don't re-embed (and re-pay for) the same document twice."""
    return hashlib.md5(file_bytes).hexdigest()

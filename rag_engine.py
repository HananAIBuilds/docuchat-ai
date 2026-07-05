"""
rag_engine.py
-------------
Core Retrieval-Augmented Generation engine.

Pipeline:
    1. Chunking      -> split raw text into overlapping windows
    2. Embedding     -> Gemini embedding model, with retry-on-failure
    3. Indexing      -> FAISS flat L2 index (in-memory vector store)
    4. Retrieval     -> top-k nearest chunks for a query
    5. Generation    -> Gemini generation model, grounded on retrieved chunks

This module has zero Streamlit imports on purpose — it can be reused in a
CLI, a notebook, or a different UI framework without modification.
"""

import time
import hashlib
import numpy as np
import faiss
from google import genai


class RAGEngine:
    EMBED_MODEL = "gemini-embedding-001"
    GEN_MODEL = "gemini-2.5-flash"

    # Tabular data (csv/xlsx) is kept in larger, less-overlapping chunks so that
    # rows / records aren't fragmented mid-way. Prose files use smaller,
    # denser chunks for tighter semantic matches.
    CHUNK_PARAMS = {
        ".csv": (4500, 250),
        ".xlsx": (4500, 250),
        "default": (800, 100),
    }

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("A Google API key is required to initialize RAGEngine.")
        self.client = genai.Client(api_key=api_key)
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
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.EMBED_MODEL,
                    contents=text,
                )
                return result.embeddings[0].values
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
             time.sleep(0.5)
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

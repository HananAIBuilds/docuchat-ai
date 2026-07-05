# 📄 DocuChat AI — Retrieval-Augmented Q&A over Your Documents

A Retrieval-Augmented Generation (RAG) app that lets you upload a document (`.txt`, `.pdf`, `.docx`, `.csv`, `.xlsx`) and ask questions about it in a chat interface. Answers are grounded strictly in the document's content, with the exact source chunks shown for every response.

**🔗 Live demo:** _add your Streamlit Cloud link here after deploying_

---

## ✨ Features

- **Multi-format support** — plain text, PDF, Word, CSV, and Excel files
- **Chat-style interface** with full conversation history
- **Transparent retrieval** — every answer shows the exact chunks it was grounded in, with similarity distance
- **Smart chunking** — tabular data (CSV/XLSX) uses larger, low-overlap chunks to keep rows intact; prose uses smaller, denser chunks for tighter semantic matches
- **Session-level caching** — a document is embedded once per session; asking follow-up questions doesn't re-embed or re-pay for the same file
- **Resilient embeddings** — automatic retry with backoff on transient API failures
- **Clear failure states** — missing API key, unsupported file type, or empty document all produce readable errors instead of crashes

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Upload File] --> B[Extract Text]
    B --> C[Chunk Text]
    C --> D[Embed Chunks<br/>gemini-embedding-001]
    D --> E[FAISS Vector Index]
    F[User Question] --> G[Embed Question]
    G --> H[Similarity Search<br/>top-k chunks]
    E --> H
    H --> I[Build Grounded Prompt]
    I --> J[Generate Answer<br/>gemini-2.5-flash]
    J --> K[Answer + Sources]
```

**Pipeline stages:**

1. **Ingestion** — `file_loader.py` extracts raw text from the uploaded file.
2. **Chunking** — the text is split into overlapping windows (size depends on file type).
3. **Embedding** — each chunk is converted into a dense vector via Gemini's embedding model.
4. **Indexing** — vectors are stored in an in-memory FAISS `IndexFlatL2` store.
5. **Retrieval** — the user's question is embedded and matched against the top-k closest chunks.
6. **Generation** — Gemini's generation model answers using only the retrieved chunks as context.

---

## 📁 Project Structure

```
rag-file-qa/
├── app.py                          # Streamlit UI (entry point)
├── rag_engine.py                   # Core RAG logic: chunk / embed / index / retrieve / generate
├── file_loader.py                  # Text extraction for txt/pdf/docx/csv/xlsx
├── requirements.txt
├── LICENSE
├── .gitignore
└── .streamlit/
    ├── config.toml                 # Theme
    └── secrets.toml.example        # Template — copy to secrets.toml, never commit the real one
```

Separating the engine from the UI means the RAG logic can be reused in a CLI tool, a notebook, or a different frontend without touching a single Streamlit call.

---

## 🚀 Running Locally

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/rag-file-qa.git
cd rag-file-qa
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API key**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your Google API key
```

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

**3. Run the app**

```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repo to GitHub (make sure `.streamlit/secrets.toml` is **not** committed — it's already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo, branch, and `app.py` as the entry point.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key-here"
   ```
4. Deploy. The app reads the key via `st.secrets`, so no code changes are needed between local and cloud environments.

---

## ⚠️ Known Limitations (read before demoing)

- **Not built for counting/aggregation.** RAG retrieves only the top-k most relevant chunks — never the whole document. A question like *"how many rows have value X?"* will only reflect the rows inside the retrieved chunks, not the full dataset. This tool is best suited for *"find this specific fact"* questions, not full-dataset statistics.
- **Processing time scales with file size.** Each chunk requires one embedding API call; very large documents take longer and cost more to process.
- **In-memory index.** The FAISS index lives in the Streamlit session — it resets if the app restarts or the session ends. There's no persistent vector database (yet).
- **Answer quality depends on retrieval quality.** If the top-k chunks don't contain the answer, the model is instructed to say so rather than guess — but retrieval isn't perfect.

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Generation | Google Gemini (`gemini-2.5-flash`) |
| Vector search | FAISS (`IndexFlatL2`) |
| File parsing | `pypdf`, `python-docx`, `openpyxl`, `csv` |

---

## 🗺️ Possible Next Steps

- Swap the in-memory FAISS index for a persistent vector DB (e.g. Chroma, Pinecone) to support multi-session use
- Add multi-file / multi-document support with per-file source attribution
- Add a lightweight aggregation path (e.g. pandas-based) for structured files so counting questions can be answered exactly, alongside the semantic RAG path
- Add automated tests around chunking edge cases and file parsing

---

## 📜 License

MIT — see [LICENSE](LICENSE).

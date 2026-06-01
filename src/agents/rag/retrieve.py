import chromadb
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

RAG_INDEX_DIR = Path("results/rag_index")
COLLECTION_NAME = "phishing_knowledge_base"
TOP_N = 3
MAX_EMAIL_CHARS = 400

_client = None
_collection = None
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(RAG_INDEX_DIR))

    existing = [c.name for c in _client.list_collections()]
    if COLLECTION_NAME in existing:
        _collection = _client.get_collection(COLLECTION_NAME)
        return _collection

    print("Building RAG index from training split — this runs once...")
    _collection = _build_index()
    return _collection


def _build_index():
    from src.evaluation.run_experiment import get_rag_dataframe

    rag_df = get_rag_dataframe()
    embedder = _get_embedder()
    collection = _client.create_collection(COLLECTION_NAME)

    texts = rag_df["text"].tolist()
    labels = rag_df["label"].tolist()
    ids = [str(i) for i in range(len(texts))]

    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]
        embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[{"label": lbl} for lbl in batch_labels],
        )

    print(f"RAG index built: {len(texts)} emails indexed.")
    return collection


def rag_retrieve(state):
    email = state["email"]
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.encode([email], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_N,
        include=["documents", "metadatas"],
    )

    examples = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = meta["label"]
        excerpt = doc[:MAX_EMAIL_CHARS].replace("\n", " ").strip()
        if len(doc) > MAX_EMAIL_CHARS:
            excerpt += "..."
        examples.append(f"[{label.upper()}]\n{excerpt}")

    rag_context = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return {"rag_context": rag_context}

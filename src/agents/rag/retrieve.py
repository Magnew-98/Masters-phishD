import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("results/rag_index")
COLLECTION = "emails"

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

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(INDEX_DIR))

    existing = [c.name for c in _client.list_collections()]
    if COLLECTION in existing:
        _collection = _client.get_collection(COLLECTION)
        return _collection

    print("Building RAG index — this runs once...")
    _collection = _build_index()
    return _collection


def _build_index():
    from src.evaluation.run_experiment import get_rag_dataframe

    df = get_rag_dataframe()
    embedder = _get_embedder()
    collection = _client.create_collection(COLLECTION)

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    ids = [str(i) for i in range(len(texts))]

    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings,
            documents=batch,
            metadatas=[{"label": l} for l in labels[i:i + batch_size]],
        )

    print(f"RAG index built: {len(texts)} emails indexed.")
    return collection


def rag_retrieve(state):
    email = state["email"]
    collection = _get_collection()
    embedder = _get_embedder()

    embedding = embedder.encode([email], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=3,
        include=["documents", "metadatas"],
    )

    examples = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        snippet = doc[:400].replace("\n", " ").strip()
        if len(doc) > 400:
            snippet += "..."
        examples.append(f"[{meta['label'].upper()}]\n{snippet}")

    context = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return {"rag_context": context}

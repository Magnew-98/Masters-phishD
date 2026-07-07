import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("results/rag_index")
COLLECTION = "emails"
NEAR_DUPLICATE_THRESHOLD = 0.10  # cosine distance; filters cosine_similarity > 0.90

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
        col = _client.get_collection(COLLECTION)
        if col.metadata and col.metadata.get("hnsw:space") == "cosine":
            _collection = col
            return _collection
        print("Rebuilding RAG index with cosine distance metric (one-time migration)...")
        _client.delete_collection(COLLECTION)

    print("Building RAG index — this runs once...")
    _collection = _build_index()
    return _collection


def _build_index():
    from src.evaluation.run_experiment import get_rag_dataframe

    df = get_rag_dataframe()
    embedder = _get_embedder()
    collection = _client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    email_ids = [int(eid) for eid in df["email_id"].tolist()]
    ids = [str(i) for i in range(len(texts))]

    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings,
            documents=batch,
            metadatas=[
                {"label": labels[i + j], "email_id": email_ids[i + j]}
                for j in range(len(batch))
            ],
        )

    print(f"RAG index built: {len(texts)} emails indexed.")
    return collection


def _query_class(collection, embedding, label: str, n_candidates: int = 5):
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_candidates,
        where={"label": label},
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def rag_retrieve(state):
    email = state["email"]
    collection = _get_collection()
    embedder = _get_embedder()

    embedding = embedder.encode([email], show_progress_bar=False).tolist()

    examples = []
    retrieved_labels = []
    retrieved_ids = []

    for label in ("phishing", "legitimate"):
        docs, metas, dists = _query_class(collection, embedding, label)
        for doc, meta, dist in zip(docs, metas, dists):
            if dist < NEAR_DUPLICATE_THRESHOLD:
                continue
            snippet = doc[:400].replace("\n", " ").strip()
            if len(doc) > 400:
                snippet += "..."
            examples.append(f"[{label.upper()}]\n{snippet}")
            retrieved_labels.append(meta["label"])
            retrieved_ids.append(str(meta["email_id"]))
            break  # one closest valid example per class

    if not examples:
        return {"rag_context": "", "rag_retrieved_labels": "", "rag_retrieved_ids": ""}

    context = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return {
        "rag_context": context,
        "rag_retrieved_labels": ",".join(retrieved_labels),
        "rag_retrieved_ids": ",".join(retrieved_ids),
    }

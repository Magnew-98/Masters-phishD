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


def rag_retrieve(state):
    email = state["email"]
    collection = _get_collection()
    embedder = _get_embedder()

    embedding = embedder.encode([email], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    examples = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist < NEAR_DUPLICATE_THRESHOLD:
            continue
        snippet = doc[:400].replace("\n", " ").strip()
        if len(doc) > 400:
            snippet += "..."
        examples.append(f"[{meta['label'].upper()}]\n{snippet}")
        if len(examples) == 3:
            break

    if not examples:
        return {"rag_context": ""}

    context = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return {"rag_context": context}

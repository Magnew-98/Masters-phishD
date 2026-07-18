import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("results/rag_index")
COLLECTION = "emails"
RAG_LOG_PATH = Path("results/rag_retrieval.log")

NEAR_DUPLICATE_THRESHOLD = 0.10  # cosine distance; filters cosine_similarity > 0.90
EXAMPLES_PER_CLASS = 2           # k=4 total (2 phishing + 2 legitimate)

_client = None
_collection = None
_embedder = None
_fixed_examples = None


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


def _query_class(collection, embedding, label: str, n_candidates: int = 15):
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_candidates,
        where={"label": label},
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def _write_log(entry: dict) -> None:
    RAG_LOG_PATH.parent.mkdir(exist_ok=True)
    with RAG_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _build_context(examples, retrieved_labels, retrieved_ids):
    if not examples:
        return {"rag_context": "", "rag_retrieved_labels": "", "rag_retrieved_ids": ""}
    context = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return {
        "rag_context": context,
        "rag_retrieved_labels": ",".join(retrieved_labels),
        "rag_retrieved_ids": ",".join(retrieved_ids),
    }


def rag_retrieve(state):
    """Stratified retrieval with near-duplicate filter (default RAG mode).
    Queries phishing and legitimate separately, filters cosine_similarity > 0.90,
    returns up to EXAMPLES_PER_CLASS from each class."""
    email = state["email"]
    email_id = state.get("email_id", "unknown")
    collection = _get_collection()
    embedder = _get_embedder()
    embedding = embedder.encode([email], show_progress_bar=False).tolist()

    examples, retrieved_labels, retrieved_ids = [], [], []
    log_classes = {}

    for label in ("phishing", "legitimate"):
        docs, metas, dists = _query_class(collection, embedding, label)
        all_candidates = [
            {"email_id": int(m["email_id"]), "dist": round(d, 4), "label": m["label"]}
            for m, d in zip(metas, dists)
        ]
        selected, count = [], 0
        for doc, meta, dist in zip(docs, metas, dists):
            if count >= EXAMPLES_PER_CLASS:
                break
            if dist < NEAR_DUPLICATE_THRESHOLD:
                continue
            snippet = doc[:400].replace("\n", " ").strip()
            if len(doc) > 400:
                snippet += "..."
            examples.append(f"[{label.upper()}]\n{snippet}")
            retrieved_labels.append(meta["label"])
            retrieved_ids.append(str(meta["email_id"]))
            selected.append({"email_id": int(meta["email_id"]), "dist": round(dist, 4)})
            count += 1
        log_classes[label] = {
            "all_candidates": all_candidates,
            "n_filtered_as_duplicate": sum(1 for c in all_candidates if c["dist"] < NEAR_DUPLICATE_THRESHOLD),
            "selected": selected,
        }

    _write_log({
        "email_id": email_id, "mode": "stratified_filtered",
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "total_retrieved": len(examples),
        "class_counts": {"phishing": retrieved_labels.count("phishing"), "legitimate": retrieved_labels.count("legitimate")},
        "classes": log_classes,
    })
    return _build_context(examples, retrieved_labels, retrieved_ids)


def rag_retrieve_nofilter(state):
    """Stratified retrieval WITHOUT near-duplicate filter.
    Same class-balanced approach but no similarity threshold applied —
    the closest examples of each class are always returned regardless of similarity."""
    email = state["email"]
    email_id = state.get("email_id", "unknown")
    collection = _get_collection()
    embedder = _get_embedder()
    embedding = embedder.encode([email], show_progress_bar=False).tolist()

    examples, retrieved_labels, retrieved_ids = [], [], []
    log_classes = {}

    for label in ("phishing", "legitimate"):
        docs, metas, dists = _query_class(collection, embedding, label)
        all_candidates = [
            {"email_id": int(m["email_id"]), "dist": round(d, 4), "label": m["label"]}
            for m, d in zip(metas, dists)
        ]
        selected, count = [], 0
        for doc, meta, dist in zip(docs, metas, dists):
            if count >= EXAMPLES_PER_CLASS:
                break
            snippet = doc[:400].replace("\n", " ").strip()
            if len(doc) > 400:
                snippet += "..."
            examples.append(f"[{label.upper()}]\n{snippet}")
            retrieved_labels.append(meta["label"])
            retrieved_ids.append(str(meta["email_id"]))
            selected.append({"email_id": int(meta["email_id"]), "dist": round(dist, 4)})
            count += 1
        log_classes[label] = {"all_candidates": all_candidates, "selected": selected}

    _write_log({
        "email_id": email_id, "mode": "stratified_nofilter",
        "total_retrieved": len(examples),
        "class_counts": {"phishing": retrieved_labels.count("phishing"), "legitimate": retrieved_labels.count("legitimate")},
        "classes": log_classes,
    })
    return _build_context(examples, retrieved_labels, retrieved_ids)


def rag_retrieve_unrestricted_filter(state):
    """Unrestricted nearest-neighbour retrieval WITH near-duplicate filter.
    Returns the top-k most similar emails regardless of class, filtering out
    cosine_similarity > 0.90 to prevent near-duplicate examples dominating."""
    email = state["email"]
    email_id = state.get("email_id", "unknown")
    collection = _get_collection()
    embedder = _get_embedder()
    embedding = embedder.encode([email], show_progress_bar=False).tolist()

    n_candidates = EXAMPLES_PER_CLASS * 2 * 4  # oversample to allow for filtering
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    examples, retrieved_labels, retrieved_ids, all_candidates = [], [], [], []
    target = EXAMPLES_PER_CLASS * 2

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        all_candidates.append({"email_id": int(meta["email_id"]), "dist": round(dist, 4), "label": meta["label"]})
        if len(examples) >= target:
            continue
        if dist < NEAR_DUPLICATE_THRESHOLD:
            continue
        snippet = doc[:400].replace("\n", " ").strip()
        if len(doc) > 400:
            snippet += "..."
        examples.append(f"[{meta['label'].upper()}]\n{snippet}")
        retrieved_labels.append(meta["label"])
        retrieved_ids.append(str(meta["email_id"]))

    _write_log({
        "email_id": email_id, "mode": "unrestricted_filter",
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "total_retrieved": len(examples),
        "class_counts": {"phishing": retrieved_labels.count("phishing"), "legitimate": retrieved_labels.count("legitimate")},
        "n_filtered": sum(1 for c in all_candidates if c["dist"] < NEAR_DUPLICATE_THRESHOLD),
        "candidates": all_candidates[:target + 5],
    })
    return _build_context(examples, retrieved_labels, retrieved_ids)


def rag_retrieve_unrestricted(state):
    """Unrestricted nearest-neighbour retrieval — no class stratification, no filter.
    Returns the top-k most similar emails regardless of class label."""
    email = state["email"]
    email_id = state.get("email_id", "unknown")
    collection = _get_collection()
    embedder = _get_embedder()
    embedding = embedder.encode([email], show_progress_bar=False).tolist()

    n_results = EXAMPLES_PER_CLASS * 2
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    examples, retrieved_labels, retrieved_ids, candidates = [], [], [], []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({"email_id": int(meta["email_id"]), "dist": round(dist, 4), "label": meta["label"]})
        snippet = doc[:400].replace("\n", " ").strip()
        if len(doc) > 400:
            snippet += "..."
        examples.append(f"[{meta['label'].upper()}]\n{snippet}")
        retrieved_labels.append(meta["label"])
        retrieved_ids.append(str(meta["email_id"]))

    _write_log({
        "email_id": email_id, "mode": "unrestricted",
        "total_retrieved": len(examples),
        "class_counts": {"phishing": retrieved_labels.count("phishing"), "legitimate": retrieved_labels.count("legitimate")},
        "candidates": candidates,
    })
    return _build_context(examples, retrieved_labels, retrieved_ids)


def _get_fixed_examples():
    global _fixed_examples
    if _fixed_examples is not None:
        return _fixed_examples

    from src.evaluation.run_experiment import get_rag_dataframe

    df = get_rag_dataframe()
    examples, retrieved_labels, retrieved_ids = [], [], []

    for label in ("phishing", "legitimate"):
        sampled = df[df["label"] == label].sample(n=EXAMPLES_PER_CLASS, random_state=98)
        for _, row in sampled.iterrows():
            doc = row["text"]
            snippet = doc[:400].replace("\n", " ").strip()
            if len(doc) > 400:
                snippet += "..."
            examples.append(f"[{label.upper()}]\n{snippet}")
            retrieved_labels.append(label)
            retrieved_ids.append(str(int(row["email_id"])))

    _fixed_examples = (examples, retrieved_labels, retrieved_ids)
    return _fixed_examples


def rag_retrieve_fixed(state):
    """Fixed few-shot: same EXAMPLES_PER_CLASS phishing + legitimate examples for every email.
    Randomly sampled from the RAG training set with random_state=98 (reproducible)."""
    examples, retrieved_labels, retrieved_ids = _get_fixed_examples()
    return _build_context(examples, retrieved_labels, retrieved_ids)

import re
import html
import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

DATASET_PATH = Path("src/datasets/Enron.csv")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "results.csv"
SPLIT_PATH = RESULTS_DIR / "split.json"

RAG_FRACTION = 0.1
RANDOM_STATE = 98


def _clean_email(text: str) -> str:
    # Decode HTML entities (&amp; &#x20; etc.)
    text = html.unescape(text)
    # Strip HTML tags, preserving text content between them
    text = re.sub(r'<[^>]+>', ' ', text)
    # Quoted-printable: join soft-wrapped lines (= at line end)
    text = re.sub(r'=\r?\n', '', text)
    # Quoted-printable: decode =XX hex sequences (=20 → space, =3D → = etc.)
    text = re.sub(r'=([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), text)
    # Normalise whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _load_full_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["email_id"] = df.index
    raw = df["subject"].fillna("") + "\n\n" + df["body"].fillna("")
    df["text"] = raw.apply(_clean_email)
    df["label"] = df["label"].map({0: "legitimate", 1: "phishing"})
    return df


def _get_split() -> tuple[list[int], list[int]]:
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text())
        return split["rag_ids"], split["test_ids"]

    RESULTS_DIR.mkdir(exist_ok=True)
    df = _load_full_dataset()
    rag_ids, test_ids = train_test_split(
        df["email_id"].tolist(),
        test_size=1 - RAG_FRACTION,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    SPLIT_PATH.write_text(json.dumps({"rag_ids": rag_ids, "test_ids": test_ids}))
    print(f"Split created: {len(rag_ids)} RAG emails, {len(test_ids)} test emails → {SPLIT_PATH}")
    return rag_ids, test_ids


def get_rag_dataframe() -> pd.DataFrame:
    rag_ids, _ = _get_split()
    df = _load_full_dataset()
    return df[df["email_id"].isin(rag_ids)].reset_index(drop=True)


def _load_results() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame(columns=["email_id", "agent_name", "true_label", "prediction", "confidence", "rag_retrieved_labels", "rag_retrieved_ids"])


LABELS = ["legitimate", "phishing"]


def _print_metrics_block(truths, preds, header: str) -> None:
    print(f"\n=== {header} ===")
    print(f"Accuracy: {accuracy_score(truths, preds):.4f}")
    print(classification_report(truths, preds, labels=LABELS, digits=4, zero_division=0))

    cm = confusion_matrix(truths, preds, labels=LABELS)
    tn, fp, fn, tp = cm.ravel()
    col_w = max(len("Predicted Legit"), len(str(max(cm.ravel())))) + 2
    print("Confusion Matrix:")
    print(f"{'':20} {'Pred Legitimate':>{col_w}} {'Pred Phishing':>{col_w}}")
    print(f"{'Actual Legitimate':20} {tn:>{col_w}} {fp:>{col_w}}")
    print(f"{'Actual Phishing':20} {fn:>{col_w}} {tp:>{col_w}}")
    print()


def print_metrics(agent_name: str) -> None:
    results = _load_results()
    agent_results = results[results["agent_name"] == agent_name]

    if agent_results.empty:
        print(f"No results found for '{agent_name}'.")
        return

    truths = agent_results["true_label"]
    preds = agent_results["prediction"]
    total = len(_get_split()[1])

    _print_metrics_block(truths, preds, f"{agent_name} | {len(agent_results)}/{total} emails evaluated")


def run(app, agent_name: str, batch_size: int = 20, dry_run: bool = False) -> None:
    if not dry_run:
        RESULTS_DIR.mkdir(exist_ok=True)

    _, test_ids = _get_split()
    df = _load_full_dataset()
    test_df = df[df["email_id"].isin(test_ids)]

    existing = _load_results()
    processed = set(existing[existing["agent_name"] == agent_name]["email_id"].tolist())
    pending = test_df[~test_df["email_id"].isin(processed)].sample(frac=1, random_state=RANDOM_STATE)

    if not dry_run and pending.empty:
        print(f"All {len(test_ids)} test emails processed for '{agent_name}'.")
        print_metrics(agent_name)
        return

    batch = pending.head(batch_size)
    remaining = len(pending) - len(batch)
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}[{agent_name}] Running {len(batch)} emails ({len(processed)} done, {remaining} remaining)")

    rows = []
    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        result = app.invoke({"email": row["text"]})
        rows.append({
            "email_id": row["email_id"],
            "agent_name": agent_name,
            "true_label": row["label"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "rag_retrieved_labels": result.get("rag_retrieved_labels", ""),
            "rag_retrieved_ids": result.get("rag_retrieved_ids", ""),
        })
        if dry_run:
            print(f"  TRUE: {row['label']}  PRED: {result['prediction']}  CONF: {result['confidence']:.2f}")

    if dry_run:
        truths = [r["true_label"] for r in rows]
        preds = [r["prediction"] for r in rows]
        _print_metrics_block(truths, preds, f"DRY RUN — {agent_name} ({len(rows)} emails)")
        print("Nothing written to disk.")
        return

    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(RESULTS_PATH, index=False)
    print(f"Appended {len(rows)} rows → {RESULTS_PATH}")
    print_metrics(agent_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phishing detection experiment batch")
    parser.add_argument("--components", required=True, nargs='+',
                        help="Agents to run: binary technical sentiment linguistic")
    parser.add_argument("--rag", action="store_true", help="Prepend RAG retrieval node")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", action="store_true",
                        help="Run specialist nodes in parallel (needs OLLAMA_NUM_PARALLEL set)")
    args = parser.parse_args()

    from src.graph.factory import build_graph, agent_name as make_name

    components = args.components
    name = make_name(components, use_rag=args.rag)

    if args.metrics_only:
        print_metrics(name)
    else:
        app = build_graph(components, use_rag=args.rag, parallel=getattr(args, "parallel", False))
        run(app, agent_name=name, batch_size=args.batch_size, dry_run=args.dry_run)

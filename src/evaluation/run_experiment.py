import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

DATASET_PATH = Path("src/datasets/Enron.csv")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "results.csv"
SPLIT_PATH = RESULTS_DIR / "split.json"

RAG_FRACTION = 0.2
RANDOM_STATE = 42


def _load_full_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["email_id"] = df.index
    df["text"] = df["subject"].fillna("") + "\n\n" + df["body"].fillna("")
    df["label"] = df["label"].map({0: "legitimate", 1: "phishing"})
    return df


def _get_split() -> tuple[list[int], list[int]]:
    """Return (rag_ids, test_ids), creating and saving the split on first call."""
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
    """Return the RAG training subset. Use this to build your RAG knowledge base."""
    rag_ids, _ = _get_split()
    df = _load_full_dataset()
    return df[df["email_id"].isin(rag_ids)].reset_index(drop=True)


def _load_results() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame(columns=["email_id", "agent_name", "true_label", "prediction", "confidence"])


def print_metrics(agent_name: str) -> None:
    results = _load_results()
    agent_results = results[results["agent_name"] == agent_name]

    if agent_results.empty:
        print(f"No results found for '{agent_name}'.")
        return

    truths = agent_results["true_label"]
    preds = agent_results["prediction"]
    total_test = len(_get_split()[1])

    print(f"\n=== {agent_name} | {len(agent_results)}/{total_test} emails evaluated ===")
    print(f"Accuracy: {accuracy_score(truths, preds):.4f}")
    print(classification_report(truths, preds, digits=4))


def run(app, agent_name: str, batch_size: int = 20, dry_run: bool = False) -> None:
    """
    Run one batch of inference for `agent_name` using the provided LangGraph `app`.

    The app must accept {"email": str} and return {"prediction": str, "confidence": float}.
    Results are appended to results/results.csv. Call repeatedly until all test emails
    are processed, or call print_metrics() to see current accumulated results.

    Set dry_run=True while building/testing agents — runs inference and prints output
    but never writes to results.csv.
    """
    if not dry_run:
        RESULTS_DIR.mkdir(exist_ok=True)

    _, test_ids = _get_split()
    df = _load_full_dataset()
    test_df = df[df["email_id"].isin(test_ids)]

    existing = _load_results()
    done_ids = set(existing[existing["agent_name"] == agent_name]["email_id"].tolist())
    pending = test_df[~test_df["email_id"].isin(done_ids)]

    if not dry_run and pending.empty:
        print(f"All {len(test_ids)} test emails already processed for '{agent_name}'.")
        print_metrics(agent_name)
        return

    batch = pending.head(batch_size)
    remaining_after = len(pending) - len(batch)
    prefix = "[DRY RUN] " if dry_run else ""
    print(
        f"{prefix}[{agent_name}] Running batch of {len(batch)} "
        f"({len(done_ids)} done, {remaining_after} remaining after this batch)"
    )

    rows = []
    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        result = app.invoke({"email": row["text"]})
        rows.append({
            "email_id": row["email_id"],
            "agent_name": agent_name,
            "true_label": row["label"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        })
        if dry_run:
            print(f"  TRUE: {row['label']}  PRED: {result['prediction']}  CONF: {result['confidence']:.2f}")

    if dry_run:
        batch_truths = [r["true_label"] for r in rows]
        batch_preds = [r["prediction"] for r in rows]
        print(f"\n[DRY RUN] Batch accuracy: {accuracy_score(batch_truths, batch_preds):.4f}")
        print(classification_report(batch_truths, batch_preds, digits=4))
        print("[DRY RUN] Nothing written to disk.")
        return

    new_results = pd.DataFrame(rows)
    updated = pd.concat([existing, new_results], ignore_index=True)
    updated.to_csv(RESULTS_PATH, index=False)

    print(f"Appended {len(rows)} rows → {RESULTS_PATH}")
    print_metrics(agent_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phishing detection experiment batch")
    parser.add_argument("--agent", required=True, help="Agent name (must match graph import)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--metrics-only", action="store_true", help="Print metrics without running inference")
    parser.add_argument("--dry-run", action="store_true", help="Run inference but do not write results to disk")
    args = parser.parse_args()

    if args.metrics_only:
        print_metrics(args.agent)
    else:
        if args.agent == "binary":
            from src.graph.binary_graph import app
            run(app, agent_name="binary", batch_size=args.batch_size, dry_run=args.dry_run)
        else:
            print(f"Unknown agent '{args.agent}'. Add it to the __main__ block or call run() directly.")

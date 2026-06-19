"""
Force-process specific email IDs that consistently fail with the normal
temperature=0.2 setting by using a higher temperature to break determinism.

Usage:
    python -m src.evaluation.force_process --agent technical_sentiment --ids 7003 8488 13422
    python -m src.evaluation.force_process --agent binary --ids 7003
"""

import argparse
import json
import time
import urllib.request
import pandas as pd
from pathlib import Path
from langchain_ollama import ChatOllama

RESULTS_PATH = Path("results/results.csv")
RANDOM_STATE = 98


def _reload_model(model: str = "llama3.1") -> None:
    try:
        data = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=30)
        time.sleep(2)
    except Exception:
        pass


def _patch_llm_temperature(temperature: float) -> None:
    """Monkey-patch get_llm to use a different temperature for this session."""
    import src.agents.shared_llm as shared
    shared._forced_temperature = temperature
    original_get_llm = shared.get_llm

    def patched_get_llm():
        return ChatOllama(
            model="llama3.1",
            temperature=temperature,
            seed=None,          # Remove seed so each call is independently random
            num_ctx=4096,
            client_kwargs={"timeout": 120},
        )

    shared.get_llm = patched_get_llm
    return original_get_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True,
                        help="Agent name, e.g. binary or technical_sentiment")
    parser.add_argument("--ids", nargs="+", type=int, required=True,
                        help="email_id values to force-process")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature to use (default 0.5 to break determinism)")
    parser.add_argument("--rag", action="store_true")
    args = parser.parse_args()

    print(f"Force-processing {len(args.ids)} email(s) for agent '{args.agent}' "
          f"with temperature={args.temperature}")

    # Patch temperature before importing agents (which call get_llm at module level)
    original_get_llm = _patch_llm_temperature(args.temperature)

    from src.evaluation.run_experiment import _load_full_dataset, _get_split, _load_results
    from src.graph.factory import build_graph, agent_name as make_name

    components = args.agent.split("_")
    app = build_graph(components, use_rag=args.rag, parallel=False)
    name = make_name(components, use_rag=args.rag)

    df = _load_full_dataset()
    existing = _load_results()
    write_header = not RESULTS_PATH.exists()

    for email_id in args.ids:
        rows = df[df["email_id"] == email_id]
        if rows.empty:
            print(f"  email_id={email_id} not found in dataset — skipping")
            continue

        already_done = (
            not existing.empty and
            len(existing[(existing["agent_name"] == name) &
                         (existing["email_id"] == email_id)]) > 0
        )
        if already_done:
            print(f"  email_id={email_id} already in results — skipping")
            continue

        row = rows.iloc[0]
        print(f"\n  Processing email_id={email_id}  chars={len(row['text'])}  label={row['label']}")

        result = None
        for attempt in range(5):
            try:
                result = app.invoke({"email": row["text"]})
                print(f"  SUCCESS on attempt {attempt + 1}  "
                      f"prediction={result.get('prediction')}  "
                      f"confidence={result.get('confidence', 0):.2f}")
                break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed: {str(e)[:80]}")
                if attempt < 4:
                    _reload_model()

        if result is None:
            print(f"  FAILED all 5 attempts — not written")
            continue

        new_row = {
            "email_id": row["email_id"],
            "agent_name": name,
            "true_label": row["label"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "rag_retrieved_labels": result.get("rag_retrieved_labels", ""),
            "rag_retrieved_ids": result.get("rag_retrieved_ids", ""),
        }
        pd.DataFrame([new_row]).to_csv(RESULTS_PATH, mode="a", header=write_header, index=False)
        write_header = False
        print(f"  Written to {RESULTS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()

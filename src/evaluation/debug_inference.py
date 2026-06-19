"""
Debug inference script — logs raw LLM responses for emails in a given
range of the shuffled test set. Saves nothing to results.csv.

Usage:
    python -m src.evaluation.debug_inference --start 10000 --count 20
    python -m src.evaluation.debug_inference --email-id 16362
    python -m src.evaluation.debug_inference --email-id 7003 8488 13422 --seed-test
    python -m src.evaluation.debug_inference --email-id 7003 --repeats 3

--seed-test  For each email, run with seed=98 (production), then 99 and 100,
             to show whether a different seed breaks the deterministic failure.
--repeats N  Run each email N times with the same seed to show determinism.
"""

import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

from src.evaluation.run_experiment import _load_full_dataset, _get_split
from src.schemas.outputs import AnalysisOutput

LOG_PATH = Path("results/debug_inference.log")
RANDOM_STATE = 98


def _build_prompt(email: str) -> str:
    return f"""
You are an expert cybersecurity analyst.

Your task is to analyse the email below for phishing indicators. Work through each category in turn before forming your overall assessment. For each category, note what you observe — including when no indicator is present, as absence of indicators is meaningful evidence of legitimacy.

Examine the email systematically across these categories:

1. URGENCY & PRESSURE — Does the email create artificial time pressure, warnings of account suspension, or consequences for inaction?
2. CREDENTIAL & DATA THEFT — Does it request login credentials, passwords, personal information, or financial details?
3. SENDER SPOOFING — Are there signs the sender identity is forged, mismatched, or impersonating a known organisation?
4. SUSPICIOUS LINKS — Are URLs present? Do any use IP addresses as hostnames, URL shorteners, lookalike domain names, or suspicious path keywords (login, verify, account, secure, update)?
5. SOCIAL ENGINEERING — Does the email exploit trust, fear, authority, urgency, or reciprocity to manipulate the recipient into taking action?
6. FINANCIAL PRESSURE — Does it involve unexpected payments, prize notifications, fines, or financial threats?
7. IMPERSONATION — Does it impersonate a bank, government agency, well-known company, or internal colleague?

Email:
{email}

Provide a concise but thorough analysis covering all seven categories. State clearly what evidence you found for each — or that nothing suspicious was observed.
"""


def _raw_ollama_call(prompt: str, seed: int = 98, timeout: int = 30) -> dict:
    """Call Ollama REST API directly — bypasses all LangChain parsing."""
    payload = json.dumps({
        "model": "llama3.1",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "seed": seed},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _structured_call(prompt: str, seed: int = 98) -> object:
    """Call via LangChain with_structured_output — same as production."""
    llm = ChatOllama(
        model="llama3.1", temperature=0.2, seed=seed,
        num_ctx=4096, client_kwargs={"timeout": 60},
    ).with_structured_output(AnalysisOutput)
    return llm.invoke(prompt)


def run_single(email_id: int, email_text: str, true_label: str,
               seed: int = 98, label: str = "") -> dict:
    prompt = _build_prompt(email_text)
    tag = f"  seed={seed}" + (f"  [{label}]" if label else "")

    print(f"\n--- Raw Ollama{tag} ---")
    record = {"email_id": email_id, "seed": seed, "label_tag": label,
              "true_label": true_label, "text_chars": len(email_text)}
    try:
        t0 = time.time()
        raw = _raw_ollama_call(prompt, seed=seed, timeout=30)
        elapsed = round(time.time() - t0, 2)
        raw_text = raw.get("response", "")
        record.update({
            "raw_response": raw_text[:200],
            "raw_chars": len(raw_text),
            "raw_elapsed_s": elapsed,
            "raw_done_reason": raw.get("done_reason", ""),
            "raw_prompt_tokens": raw.get("prompt_eval_count"),
            "raw_output_tokens": raw.get("eval_count"),
        })
        print(f"  elapsed={elapsed}s  tokens_in={raw.get('prompt_eval_count')}  "
              f"tokens_out={raw.get('eval_count')}  done={raw.get('done_reason')}")
        print(f"  response: {raw_text[:150]}")
    except Exception as e:
        record["raw_error"] = str(e)
        print(f"  ERROR: {e}")

    print(f"\n--- Structured LangChain{tag} ---")
    try:
        t0 = time.time()
        result = _structured_call(prompt, seed=seed)
        elapsed = round(time.time() - t0, 2)
        record.update({
            "struct_success": True,
            "struct_leaning": result.leaning,
            "struct_analysis_chars": len(result.analysis),
            "struct_elapsed_s": elapsed,
        })
        print(f"  SUCCESS  leaning={result.leaning}  analysis_chars={len(result.analysis)}  elapsed={elapsed}s")
    except Exception as e:
        record["struct_success"] = False
        record["struct_error"] = str(e)[:120]
        print(f"  FAILED: {str(e)[:120]}")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--email-id", nargs="+", type=int, default=None)
    parser.add_argument("--seed-test", action="store_true",
                        help="Test each email with seeds 98, 99, 100 to check determinism")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each email N times with seed=98 to confirm determinism")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(exist_ok=True)

    _, test_ids = _get_split()
    df = _load_full_dataset()
    test_df = df[df["email_id"].isin(test_ids)]
    shuffled = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    if args.email_id is not None:
        sample = df[df["email_id"].isin(args.email_id)]
    else:
        end = min(args.start + args.count, len(shuffled))
        sample = shuffled.iloc[args.start:end]
        print(f"Testing shuffled positions {args.start}–{end-1} ({len(sample)} emails)")

    records = []
    with open(LOG_PATH, "w") as log_file:
        for _, row in sample.iterrows():
            eid = int(row["email_id"])
            print(f"\n{'='*70}")
            print(f"email_id={eid}  label={row['label']}  chars={len(row['text'])}")
            print(f"preview: {row['text'][:120].replace(chr(10),' ')}")

            seeds_to_test = []
            if args.seed_test:
                seeds_to_test = [(98, "production"), (99, "retry-seed-99"), (100, "retry-seed-100")]
            else:
                seeds_to_test = [(98, "")] * args.repeats

            for seed, label in seeds_to_test:
                rec = run_single(eid, row["text"], row["label"], seed=seed, label=label)
                records.append(rec)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()

    failures = [r for r in records if not r.get("struct_success", False)]
    successes_by_seed = {}
    for r in records:
        s = r["seed"]
        successes_by_seed.setdefault(s, {"ok": 0, "fail": 0})
        if r.get("struct_success"):
            successes_by_seed[s]["ok"] += 1
        else:
            successes_by_seed[s]["fail"] += 1

    print(f"\n{'='*70}")
    print(f"Summary: {len(records)} calls, {len(failures)} failures")
    print("Results by seed:")
    for seed, counts in sorted(successes_by_seed.items()):
        print(f"  seed={seed}: {counts['ok']} ok, {counts['fail']} fail")
    print(f"\nLog: {LOG_PATH}")


if __name__ == "__main__":
    main()

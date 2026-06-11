"""
Debug inference script — logs raw LLM responses for emails in a given
range of the shuffled test set. Saves nothing to results.csv.

Usage:
    python -m src.evaluation.debug_inference --start 10000 --count 20
    python -m src.evaluation.debug_inference --start 10000 --count 20 --email-id 16362

The --start / --count arguments select emails by their position in the
shuffled evaluation order (same order used by run_experiment.py).
Use --email-id to test one specific email regardless of position.
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


def _raw_ollama_call(prompt: str) -> dict:
    """Call Ollama REST API directly — bypasses all LangChain parsing."""
    payload = json.dumps({
        "model": "llama3.1",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "seed": 98},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _structured_call(prompt: str):
    """Call via LangChain with_structured_output — same as production."""
    llm = ChatOllama(model="llama3.1", temperature=0.2, seed=98).with_structured_output(AnalysisOutput)
    return llm.invoke(prompt)


def debug_email(email_id: int, email_text: str, true_label: str) -> dict:
    prompt = _build_prompt(email_text)
    record = {
        "email_id": email_id,
        "true_label": true_label,
        "text_chars": len(email_text),
        "text_preview": email_text[:300].replace("\n", " "),
        "prompt_chars": len(prompt),
    }

    print(f"\n{'='*70}")
    print(f"email_id={email_id}  label={true_label}  chars={len(email_text)}")
    print(f"text preview: {email_text[:150].replace(chr(10),' ')}")

    # --- Raw Ollama call (no LangChain parsing) ---
    print("\n[RAW OLLAMA CALL]")
    try:
        t0 = time.time()
        raw = _raw_ollama_call(prompt)
        elapsed = round(time.time() - t0, 2)
        raw_text = raw.get("response", "")
        record["raw_response"] = raw_text
        record["raw_response_chars"] = len(raw_text)
        record["raw_elapsed_s"] = elapsed
        record["raw_done_reason"] = raw.get("done_reason", "")
        record["raw_prompt_tokens"] = raw.get("prompt_eval_count", None)
        record["raw_output_tokens"] = raw.get("eval_count", None)
        print(f"  elapsed: {elapsed}s")
        print(f"  done_reason: {raw.get('done_reason')}")
        print(f"  prompt_tokens: {raw.get('prompt_eval_count')}  output_tokens: {raw.get('eval_count')}")
        print(f"  raw response ({len(raw_text)} chars):\n  {raw_text[:500]}")
    except Exception as e:
        record["raw_error"] = str(e)
        print(f"  ERROR: {e}")

    # --- Structured LangChain call (same as production) ---
    print("\n[STRUCTURED LANGCHAIN CALL]")
    try:
        t0 = time.time()
        result = _structured_call(prompt)
        elapsed = round(time.time() - t0, 2)
        record["struct_success"] = True
        record["struct_leaning"] = result.leaning
        record["struct_analysis_chars"] = len(result.analysis)
        record["struct_elapsed_s"] = elapsed
        print(f"  SUCCESS  leaning={result.leaning}  analysis_chars={len(result.analysis)}  elapsed={elapsed}s")
    except Exception as e:
        record["struct_success"] = False
        record["struct_error"] = str(e)
        print(f"  FAILED: {e}")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10000,
                        help="Start position in shuffled test order")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of emails to test")
    parser.add_argument("--email-id", type=int, default=None,
                        help="Test a specific email_id only")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(exist_ok=True)

    _, test_ids = _get_split()
    df = _load_full_dataset()
    test_df = df[df["email_id"].isin(test_ids)]
    shuffled = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    if args.email_id is not None:
        rows = df[df["email_id"] == args.email_id]
        if rows.empty:
            print(f"email_id {args.email_id} not found")
            sys.exit(1)
        sample = rows
    else:
        end = min(args.start + args.count, len(shuffled))
        sample = shuffled.iloc[args.start:end]
        print(f"Testing emails at shuffled positions {args.start}–{end-1} ({len(sample)} emails)")

    records = []
    write_header = True
    with open(LOG_PATH, "w") as log_file:
        for _, row in sample.iterrows():
            rec = debug_email(int(row["email_id"]), row["text"], row["label"])
            records.append(rec)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()

    failures = [r for r in records if not r.get("struct_success", False)]
    print(f"\n{'='*70}")
    print(f"Summary: {len(records)} tested, {len(failures)} structured failures")
    for r in failures:
        print(f"  FAILED email_id={r['email_id']}  raw_chars={r.get('raw_response_chars','?')}  error={r.get('struct_error','?')[:100]}")
    print(f"\nFull log: {LOG_PATH}")


if __name__ == "__main__":
    main()

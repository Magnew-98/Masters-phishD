"""
Debug inference script — logs raw LLM responses for specific emails.
Saves nothing to results.csv.

Usage:
    # Test specific emails against all specialist prompts
    python -m src.evaluation.debug_inference --email-id 7003 8488 13422 --prompt all

    # Run the full technical_sentiment pipeline and catch which node fails
    python -m src.evaluation.debug_inference --email-id 7003 8488 13422 --pipeline technical_sentiment

    # Test determinism: run 3 times with seed=98
    python -m src.evaluation.debug_inference --email-id 7003 --repeats 3

    # Test if a different seed fixes it
    python -m src.evaluation.debug_inference --email-id 7003 --seed-test

    # Scan a range of the shuffled test set
    python -m src.evaluation.debug_inference --start 10000 --count 20
"""

import sys
import json
import time
import traceback
import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

from src.evaluation.run_experiment import _load_full_dataset, _get_split
from src.schemas.outputs import AnalysisOutput, ClassificationOutput

LOG_PATH = Path("results/debug_inference.log")
RANDOM_STATE = 98
SLOW_THRESHOLD = 15  # seconds — flag calls slower than this


# ── Prompt builders (exact copies of production prompts) ─────────────────────

def _prompt_binary(email):
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


def _prompt_technical(email):
    return f"""
You are a technical cybersecurity analyst specialising in email infrastructure threats, with deep expertise in URL analysis, domain spoofing detection, and malicious payload identification.

Your task is to examine the email below for technical indicators of phishing. Work through each category in turn. For every category, state what you found — or explicitly note that nothing suspicious was observed. Absence of technical indicators is meaningful evidence.

Examine each category systematically:

1. URLs & LINKS — Identify all URLs present. For each, assess:
   - Is the hostname an IP address rather than a domain? (strong indicator)
   - Is a URL shortener used, obscuring the real destination? (strong indicator)
   - Does the path or subdomain contain suspicious keywords: login, verify, account, secure, update, confirm, password, signin? (moderate indicator)
   - Is the TLD unusual or associated with abuse: .xyz, .tk, .ml, .ga, .cf, .click, .top? (moderate indicator)
   - Is the subdomain chain unusually deep (4+ levels)? (moderate indicator)

2. DOMAIN SPOOFING — Look for lookalike domains that impersonate legitimate brands using:
   - Character substitution (paypa1.com, g00gle.com, rnicrosft.com)
   - Added words (amazon-secure.com, paypal-login.net)
   - Different TLD on a known brand (apple.co vs apple.com)

3. SENDER & REPLY-TO — Are From and Reply-To addresses both present? Do they match? Does the sender domain correspond to the claimed organisation?

4. FILE ATTACHMENTS — Are there references to executable or high-risk file types: .exe, .zip, .js, .vbs, .bat, .cmd, .ps1, .jar, .docm, .xlsm? Are there calls to action to open, download, or run a file?

5. EMBEDDED CONTENT — Is there HTML markup, base64-encoded content, or script references embedded in what should be a plain-text email?

Email:
{email}

Provide a concise technical analysis covering all five categories. Do not classify the email — only report and interpret the technical evidence found.

Based on your analysis, provide your overall leaning: "phishing" if you found meaningful technical indicators suggesting deceptive infrastructure, "legitimate" if the email shows no suspicious technical features, or "uncertain" if the evidence is mixed or ambiguous.
"""


def _prompt_sentiment(email):
    return f"""
You are a social engineering specialist with experience identifying phishing campaigns that exploit psychological vulnerabilities.

Your task is to analyse the email below for psychological manipulation. Focus only on indicators that are genuinely distinctive of phishing — not features that are common in normal business communication.

Key distinction: urgency, authority, deadlines, and consequences are routine in corporate email and are NOT reliable phishing indicators on their own. You are looking for manipulation that would be implausible or out of place in a legitimate professional context.

Examine only these three categories — they are the most distinctive for phishing:

1. REWARD & UNSOLICITED GAIN — Does the email offer unexpected money, prizes, lottery winnings, unclaimed funds, or implausible financial benefits that require personal action to claim?

2. IMPLAUSIBLE EXTERNAL AUTHORITY — Does the email claim authority from an organisation that would have no legitimate reason to contact this recipient in this way?

3. EXTREME ARTIFICIAL URGENCY — Is there a specific, implausible countdown or ultimatum that serves no legitimate business purpose?

Email:
{email}

Provide a concise analysis of the three categories. Do not classify the email — only report the affective evidence.

Based on your analysis, provide your overall leaning: "phishing" if you found a clearly implausible manipulation tactic, "legitimate" if all emotional features are consistent with normal business communication, or "uncertain" if you are unsure.
"""


PROMPT_BUILDERS = {
    "binary": _prompt_binary,
    "technical": _prompt_technical,
    "sentiment": _prompt_sentiment,
}


# ── Raw Ollama call ──────────────────────────────────────────────────────────

def _raw_call(prompt: str, seed: int = 98, timeout: int = 30) -> dict:
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


def _structured_call(prompt: str, schema, seed: int = 98) -> object:
    llm = ChatOllama(
        model="llama3.1", temperature=0.2, seed=seed,
        num_ctx=4096, client_kwargs={"timeout": 60},
    ).with_structured_output(schema)
    return llm.invoke(prompt)


# ── Per-prompt test ──────────────────────────────────────────────────────────

def test_prompt(email_id: int, email_text: str, true_label: str,
                prompt_type: str, seed: int = 98) -> dict:
    build = PROMPT_BUILDERS[prompt_type]
    schema = ClassificationOutput if prompt_type == "classify" else AnalysisOutput
    prompt = build(email_text)
    record = {
        "email_id": email_id, "true_label": true_label,
        "prompt_type": prompt_type, "seed": seed,
        "text_chars": len(email_text), "prompt_chars": len(prompt),
    }

    print(f"\n  [{prompt_type}  seed={seed}]")

    # Raw call
    try:
        t0 = time.time()
        raw = _raw_call(prompt, seed=seed, timeout=30)
        elapsed = round(time.time() - t0, 2)
        raw_text = raw.get("response", "")
        slow = "  *** SLOW ***" if elapsed > SLOW_THRESHOLD else ""
        record.update({
            "raw_response_full": raw_text,
            "raw_chars": len(raw_text),
            "raw_elapsed_s": elapsed,
            "raw_done_reason": raw.get("done_reason", ""),
            "raw_prompt_tokens": raw.get("prompt_eval_count"),
            "raw_output_tokens": raw.get("eval_count"),
        })
        print(f"  Raw:  {elapsed}s{slow}  tokens_in={raw.get('prompt_eval_count')}  "
              f"tokens_out={raw.get('eval_count')}  done={raw.get('done_reason')}")
        print(f"  Raw response: {repr(raw_text[:300])}")
    except Exception as e:
        record["raw_error"] = str(e)
        print(f"  Raw ERROR: {e}")

    # Structured call
    try:
        t0 = time.time()
        result = _structured_call(prompt, AnalysisOutput, seed=seed)
        elapsed = round(time.time() - t0, 2)
        slow = "  *** SLOW ***" if elapsed > SLOW_THRESHOLD else ""
        record.update({
            "struct_success": True,
            "struct_leaning": result.leaning,
            "struct_analysis": result.analysis,  # full analysis text
            "struct_analysis_chars": len(result.analysis),
            "struct_elapsed_s": elapsed,
        })
        print(f"  Structured:  {elapsed}s{slow}  leaning={result.leaning}  "
              f"analysis_chars={len(result.analysis)}")
        print(f"  Analysis text: {repr(result.analysis[:400])}")
    except Exception as e:
        record["struct_success"] = False
        record["struct_error"] = str(e)
        record["struct_traceback"] = traceback.format_exc()
        print(f"  Structured FAILED: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")

    return record


# ── Full pipeline test ───────────────────────────────────────────────────────

def test_pipeline(email_id: int, email_text: str, true_label: str,
                  components: list[str], use_rag: bool = False) -> dict:
    from src.graph.factory import build_graph, agent_name as make_name
    app = build_graph(components, use_rag=use_rag, parallel=False)
    name = make_name(components, use_rag=use_rag)

    print(f"\n  [FULL PIPELINE: {name}]")
    record = {
        "email_id": email_id, "true_label": true_label,
        "pipeline": name, "text_chars": len(email_text),
    }
    try:
        t0 = time.time()
        result = app.invoke({"email": email_text})
        elapsed = round(time.time() - t0, 2)
        record.update({
            "pipeline_success": True,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "elapsed_s": elapsed,
            "technical_analysis": result.get("technical_analysis", "")[:300],
            "sentiment_analysis": result.get("sentiment_analysis", "")[:300],
            "technical_leaning": result.get("technical_leaning"),
            "sentiment_leaning": result.get("sentiment_leaning"),
        })
        print(f"  SUCCESS  {elapsed}s  prediction={result.get('prediction')}  "
              f"confidence={result.get('confidence', 0):.2f}")
        print(f"  Technical leaning: {result.get('technical_leaning')}  "
              f"analysis: {repr(result.get('technical_analysis','')[:150])}")
        print(f"  Sentiment leaning: {result.get('sentiment_leaning')}  "
              f"analysis: {repr(result.get('sentiment_analysis','')[:150])}")
    except Exception as e:
        record.update({
            "pipeline_success": False,
            "pipeline_error": str(e),
            "pipeline_traceback": traceback.format_exc(),
        })
        print(f"  PIPELINE FAILED: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")

    return record


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--email-id", nargs="+", type=int, default=None)
    parser.add_argument("--seed-test", action="store_true",
                        help="Test with seeds 98, 99, 100")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each test N times with seed=98")
    parser.add_argument("--prompt", choices=["binary", "technical", "sentiment", "all"],
                        default="all",
                        help="Which prompt(s) to test (default: all)")
    parser.add_argument("--pipeline",
                        help="Also run the full named pipeline, e.g. technical_sentiment")
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

    prompt_types = (["binary", "technical", "sentiment"]
                    if args.prompt == "all" else [args.prompt])

    seeds = ([(98, "seed-98"), (99, "seed-99"), (100, "seed-100")]
             if args.seed_test else [(98, "")] * args.repeats)

    records = []
    with open(LOG_PATH, "w") as log_file:
        for _, row in sample.iterrows():
            eid = int(row["email_id"])
            print(f"\n{'='*70}")
            print(f"email_id={eid}  label={row['label']}  chars={len(row['text'])}")
            print(f"preview: {row['text'][:120].replace(chr(10),' ')}")

            for pt in prompt_types:
                print(f"\n  ~~~ Prompt: {pt} ~~~")
                for seed, lbl in seeds:
                    rec = test_prompt(eid, row["text"], row["label"], pt, seed=seed)
                    records.append(rec)
                    log_file.write(json.dumps(rec) + "\n")
                    log_file.flush()

            if args.pipeline:
                components = args.pipeline.split("_")
                rec = test_pipeline(eid, row["text"], row["label"], components)
                records.append(rec)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()

    # Summary
    print(f"\n{'='*70}")
    print(f"Total calls: {len(records)}")
    by_prompt = {}
    for r in records:
        pt = r.get("prompt_type", r.get("pipeline", "pipeline"))
        by_prompt.setdefault(pt, {"ok": 0, "fail": 0})
        if r.get("struct_success") or r.get("pipeline_success"):
            by_prompt[pt]["ok"] += 1
        else:
            by_prompt[pt]["fail"] += 1
    for pt, counts in by_prompt.items():
        print(f"  {pt}: {counts['ok']} ok  {counts['fail']} fail")
    print(f"\nFull log: {LOG_PATH}")


if __name__ == "__main__":
    main()

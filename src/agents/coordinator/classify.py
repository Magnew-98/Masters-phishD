from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def coordinate(state):
    leaning_lines = []
    sections = []

    if state.get("analysis"):
        leaning_lines.append(f"  General analyst leaning:    {state.get('analysis_leaning', 'uncertain').upper()}")
        sections.append(f"GENERAL ANALYSIS:\n{state['analysis']}")
    if state.get("technical_analysis"):
        leaning_lines.append(f"  Technical analyst leaning:  {state.get('technical_leaning', 'uncertain').upper()}")
        sections.append(f"TECHNICAL ANALYSIS:\n{state['technical_analysis']}")
    if state.get("sentiment_analysis"):
        leaning_lines.append(f"  Sentiment analyst leaning:  {state.get('sentiment_leaning', 'uncertain').upper()}")
        sections.append(f"SENTIMENT ANALYSIS:\n{state['sentiment_analysis']}")
    if state.get("linguistic_analysis"):
        leaning_lines.append(f"  Linguistic analyst leaning: {state.get('linguistic_leaning', 'uncertain').upper()}")
        sections.append(f"LINGUISTIC ANALYSIS:\n{state['linguistic_analysis']}")

    leaning_summary = "\n".join(leaning_lines)
    combined = "\n\n---\n\n".join(sections)
    rag_context = state.get("rag_context") or ""

    rag_block = ""
    if rag_context:
        rag_block = f"""
The following are similar emails retrieved from a knowledge base of previously classified emails.
Use them as reference points to inform your decision, but base your final classification on the
analyses above.

{rag_context}

---
"""

    prompt = f"""
You are the Head of Threat Analysis at a Security Operations Centre, responsible for making final email disposition decisions based on multi-source intelligence reports from specialist analysts. You have over 15 years of experience distinguishing sophisticated phishing campaigns from legitimate business communication.

Your task is to synthesise the specialist analyses below into a single, well-reasoned classification. You are not summarising — you are forming your own independent judgement based on the weight and quality of the evidence presented.
{rag_block}
Specialist leanings (each analyst's directional conclusion from their domain):
{leaning_summary}

Full specialist analyses:

{combined}

Reasoning process — work through these steps before concluding:

Step 1 — LEANING SUMMARY: Start with the specialist leanings above. Count how many lean phishing, legitimate, or uncertain. The leanings are your primary signal — they reflect each analyst's own conclusion, not just the presence of any indicator.

Step 2 — EVIDENCE QUALITY: For any analyst leaning toward phishing, assess whether their evidence is strong (specific, objective, rare in legitimate email) or weak (context-dependent, common in corporate email). Apply these weights:
  - Technical indicators (spoofed domains, IP URLs, dangerous attachments) — strong, objective evidence
  - Linguistic indicators (homoglyphs, deliberate misspellings) — strong when present
  - Sentiment indicators (implausible reward, extreme urgency) — moderate; only strong when clearly disproportionate
  - General analysis — useful context, lower specificity

Step 3 — STEP BACK: Before concluding, ask: what is the single most specific and compelling piece of evidence in either direction? Would a skilled analyst find this email suspicious based on concrete facts, or only based on tone and phrasing?

Step 4 — CLASSIFY: Based on the leanings and evidence quality, make your determination.

Classification definitions:
- PHISHING: Deliberate deception — impersonating trusted entities, credential harvesting, manufactured urgency, malware distribution, or psychological manipulation under false pretences
- LEGITIMATE: Normal business or personal communication — may contain links, urgency, or requests but lacks deceptive intent

Calibration guidance:
- All analysts lean legitimate → legitimate, high confidence (0.85+)
- All analysts lean phishing with strong evidence → phishing, high confidence (0.85+)
- Majority lean phishing with at least one strong indicator → phishing, moderate confidence (0.70–0.85)
- Majority lean legitimate with one weak phishing signal → legitimate, moderate confidence (0.65–0.80)
- Mixed leanings with no strong indicator → uncertain, classify toward most specific evidence (0.55–0.65)
- "Uncertain" leanings from analysts should not be treated as phishing votes

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

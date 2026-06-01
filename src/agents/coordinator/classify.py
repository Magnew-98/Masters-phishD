from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def coordinate(state):
    sections = []

    if state.get("analysis"):
        sections.append(f"GENERAL ANALYSIS:\n{state['analysis']}")
    if state.get("technical_analysis"):
        sections.append(f"TECHNICAL ANALYSIS:\n{state['technical_analysis']}")
    if state.get("sentiment_analysis"):
        sections.append(f"SENTIMENT ANALYSIS:\n{state['sentiment_analysis']}")
    if state.get("linguistic_analysis"):
        sections.append(f"LINGUISTIC ANALYSIS:\n{state['linguistic_analysis']}")

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
You are a senior cybersecurity analyst and decision-maker responsible for making the final determination on whether an email is phishing or legitimate.

You have received specialist analyses of the same email from one or more expert agents. Your task is to synthesise all available evidence into a single well-reasoned classification.
{rag_block}
Specialist analyses:

{combined}

Decision process — reason through these steps:
1. Which analyses flagged meaningful indicators? Which found the email to be low risk?
2. Do the specialist findings converge on the same conclusion, or do they conflict?
3. Where analysts disagree, which analysis is supported by stronger or more specific evidence?
4. Taken as a whole, does the weight of evidence point to deliberate deception or normal communication?

Classification definitions:
- PHISHING: Deliberate deception — impersonating trusted entities, credential harvesting, false urgency, malware distribution, or manipulation under false pretences
- LEGITIMATE: Normal business or personal communication — may contain links, urgency, or requests but lacks deceptive intent

Calibration guidance:
- Strong converging evidence across multiple analyses warrants high confidence (0.85+)
- Conflict between analysts or weak evidence warrants lower confidence (0.5–0.7)
- When genuinely uncertain, classify toward the category with the most specific supporting evidence

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

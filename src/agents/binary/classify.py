from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_email(state):
    analysis = state["analysis"]

    prompt = f"""
You are a cybersecurity classification expert responsible for making accurate, well-reasoned binary decisions on whether an email is phishing or legitimate.

Your task is to read the analysis below and make a single classification.

Classification definitions:
- LEGITIMATE: Normal business communication, internal discussion, transactional emails, routine requests. May contain links, urgency, or requests but lacks deliberate deceptive intent.
- PHISHING: Emails that deliberately deceive the recipient — impersonating trusted entities, harvesting credentials or personal data, creating false urgency to force action, distributing malware, or manipulating behaviour under false pretences.

Decision process — reason through these steps:
1. How many phishing indicator categories were flagged in the analysis?
2. Are the flagged indicators strong (e.g. spoofed domain, credential request) or weak (e.g. mild urgency in a business email)?
3. Do multiple indicators converge on the same deceptive goal, or are they isolated and explainable by normal business context?
4. Does the overall pattern match a known phishing tactic, or is it consistent with routine communication?

Calibration guidance:
- A single mild indicator does not make an email phishing
- Multiple converging strong indicators justify a phishing classification
- When genuinely uncertain, classify toward the category supported by the weight of evidence
- Confidence should reflect your certainty: 0.5 = uncertain, 0.7 = moderately confident, 0.9+ = strong evidence

Analysis:
{analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)

    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }
from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_technical(state):
    technical_analysis = state["technical_analysis"]

    prompt = f"""
You are a cybersecurity classification expert evaluating technical email evidence to make a binary phishing or legitimate determination.

Your task is to read the technical analysis below and make a single classification based only on the technical evidence — not sentiment, tone, or language quality.

Classification definitions:
- LEGITIMATE: No meaningful technical red flags, or only low-risk indicators that are common in normal email (e.g. a plain URL with no suspicious keywords or unusual TLD).
- PHISHING: Technical indicators that reveal deliberate infrastructure deception — IP-as-hostname URLs, spoofed or lookalike domains, URL shorteners obscuring destinations, dangerous file attachments, or mismatched sender/reply-to addresses.

Decision process — reason through these steps:
1. Which technical indicator categories were flagged?
2. Are the flagged indicators strong (IP hostname, domain spoofing, mismatched reply-to) or weak (a single plain URL)?
3. Do multiple indicators combine to suggest a coordinated phishing infrastructure?
4. Is the absence of indicators consistent with a legitimate email?

Calibration guidance:
- One strong indicator alone (e.g. spoofed domain) is sufficient to classify as phishing
- One weak indicator alone should lean legitimate
- Multiple weak indicators together may justify phishing
- No indicators found should classify as legitimate with high confidence
- Confidence should reflect certainty: 0.5 = uncertain, 0.7 = moderately confident, 0.9+ = strong technical evidence

Technical analysis:
{technical_analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }

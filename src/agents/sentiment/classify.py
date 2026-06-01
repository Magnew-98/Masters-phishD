from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_sentiment(state):
    sentiment_analysis = state["sentiment_analysis"]

    prompt = f"""
You are a cybersecurity classification expert specialising in social engineering detection, with deep experience distinguishing manufactured psychological manipulation from normal business communication.

Your task is to classify the email as phishing or legitimate based solely on the psychological evidence in the sentiment analysis below. Reason through the evidence step by step before reaching your conclusion.

IMPORTANT CALIBRATION: Sentiment indicators are inherently weaker signals than technical ones. Urgency, authority, and time pressure are routine in professional email. Only classify as phishing when the analysis identifies manipulation that is clearly disproportionate to any plausible business context.

Classification definitions:
- LEGITIMATE: Normal business or personal communication. Routine urgency, authority, deadlines, and requests are expected and not indicative of phishing.
- PHISHING: Email where psychological manipulation is deliberately engineered to exploit emotional vulnerabilities — manufactured fear disconnected from reality, impersonated authority making implausible requests, artificial urgency designed to prevent rational evaluation, or fabricated trust.

Decision process — reason through each step before concluding:
1. Did the analysis identify manipulation explicitly flagged as disproportionate or suspicious — not merely present?
2. Do multiple manipulation tactics converge on the same goal of overriding rational judgement?
3. Is the emotional pressure proportionate to a plausible business situation, or does it only make sense as a deliberate attempt to deceive?
4. Step back: if a legitimate businessperson sent this email, which elements would naturally be present? What can only be explained by deceptive intent?

Calibration guidance — high bar required for phishing:
- Routine urgency, authority, or deadline language alone = legitimate
- Single suspicious indicator without corroboration = lean legitimate, low confidence
- Two or more converging disproportionate manipulation tactics = phishing, moderate confidence
- Clear manufactured fear or false authority impersonation = phishing, high confidence
- No suspicious manipulation found = legitimate, high confidence
- Confidence: 0.5 = uncertain, 0.7 = moderate evidence, 0.9+ = strong converging evidence

Sentiment analysis:
{sentiment_analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

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
1. Were any of the three distinctive indicators (reward/gain, implausible authority, extreme urgency) flagged as suspicious — not just present?
2. Step back: could every element of this email's emotional appeal be explained by a legitimate business sender? If yes, classify as legitimate.
3. Only if at least one indicator is clearly implausible in a business context should you lean toward phishing.

Calibration guidance — the bar for phishing is high:
- No distinctive indicators found = legitimate, high confidence (0.85+)
- Routine business urgency, authority, or deadlines only = legitimate
- One clearly implausible indicator = phishing, moderate confidence (0.65–0.75)
- Two or more clearly implausible indicators = phishing, high confidence (0.80+)
- Uncertainty about whether an indicator is genuine or manufactured = lean legitimate (0.55–0.65)
- Confidence: 0.5 = genuinely uncertain, 0.7 = moderate evidence, 0.9+ = unambiguous manipulation

Sentiment analysis:
{sentiment_analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

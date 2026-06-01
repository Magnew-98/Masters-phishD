from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_sentiment(state):
    sentiment_analysis = state["sentiment_analysis"]

    prompt = f"""
You are a cybersecurity classification expert evaluating psychological and affective email evidence to make a binary phishing or legitimate determination.

Your task is to read the sentiment analysis below and classify the email based solely on the psychological manipulation evidence — not technical indicators or linguistic quality.

Classification definitions:
- LEGITIMATE: Normal business or personal communication. May contain some urgency or authority (e.g. a manager chasing a deadline) but lacks deliberate psychological manipulation designed to bypass rational judgement.
- PHISHING: Email that deliberately exploits cognitive biases or emotional responses — manufactured fear, false authority, artificial urgency, or deceptive appeals to trust — to manipulate the recipient into taking a harmful action.

Decision process — reason through these steps:
1. Which manipulation categories were flagged in the analysis?
2. Are the tactics isolated and explainable by normal business context, or do they form a deliberate pattern designed to override careful thinking?
3. Is the emotional pressure proportionate to a genuine situation, or disproportionate in a way that suggests manufactured manipulation?
4. Do multiple tactics converge on the same goal (e.g. fear + authority + urgency all pushing toward the same action)?

Calibration guidance:
- A single mild urgency cue in a business context should lean legitimate
- Deliberate fear combined with authority impersonation is a strong phishing signal
- Multiple converging manipulation tactics justify phishing with high confidence
- Absence of manipulation tactics should classify as legitimate with high confidence
- Confidence should reflect certainty: 0.5 = uncertain, 0.7 = moderately confident, 0.9+ = strong evidence

Sentiment analysis:
{sentiment_analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

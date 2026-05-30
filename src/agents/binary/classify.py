from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_email(state):
    analysis = state["analysis"]

    prompt = f"""
You are a cybersecurity classification system.

Based on the analysis below, classify the email as EITHER phishing OR legitimate.
Normal business emails, internal communications, and routine messages are legitimate even if they contain urgency or requests.
Only classify as phishing if there is clear evidence of deliberate deception intended to steal credentials, money, or personal information.

Analysis:
{analysis}

Classify as phishing only if the analysis identifies concrete deceptive intent, not just because the email contains urgency or a request.
"""

    result = llm.invoke(prompt)

    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }
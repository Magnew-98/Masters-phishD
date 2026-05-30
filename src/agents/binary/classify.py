from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_email(state):
    analysis = state["analysis"]

    prompt = f"""
You are a cybersecurity classification system.

Based on the analysis below, classify the email.

Rules:
- phishing = any deception, credential theft, urgency manipulation
- legitimate = normal communication

Analysis:
{analysis}
"""

    result = llm.invoke(prompt)

    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }
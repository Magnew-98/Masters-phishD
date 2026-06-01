from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_technical(state):
    technical_analysis = state["technical_analysis"]

    prompt = f"""
You are a cybersecurity classification system.

Based solely on the technical analysis below, classify the email as phishing or legitimate.
Only classify as phishing if the technical evidence indicates deliberate deceptive intent —
for example, spoofed domains, IP-as-hostname URLs, or dangerous file attachments.
Absence of technical indicators should lean toward legitimate.

Technical analysis:
{technical_analysis}
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }

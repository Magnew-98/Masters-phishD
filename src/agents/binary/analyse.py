from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_email(state):
    email = state["email"]

    prompt = f"""
You are a cybersecurity analyst.

Analyse the email for phishing indicators.

Focus on:
- urgency
- credential theft
- spoofing
- suspicious links
- social engineering
- financial pressure
- impersonation

Email:
{email}

Return a structured analysis.
"""

    result = llm.invoke(prompt)

    return {
        "analysis": result.analysis
    }
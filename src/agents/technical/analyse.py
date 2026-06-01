from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_technical(state):
    email = state["email"]

    prompt = f"""
You are a technical cybersecurity analyst specialising in email infrastructure threats.

Examine the email below and extract any technical indicators that may suggest phishing.

Focus on:
- URLs: IP addresses used as hostnames, URL shorteners, suspicious keywords (login, verify, account, secure), unusual TLDs, deep subdomain chains
- Domain spoofing: slight misspellings or character substitutions in domain names (e.g. paypa1.com, g00gle.com)
- Mismatched or suspicious sender addresses and reply-to fields
- References to file attachments with dangerous extensions (.exe, .zip, .js, .vbs, .bat, .docm)
- Calls to action that involve downloading or opening files
- Embedded HTML, scripts, or encoded content

Email:
{email}

Provide a concise technical analysis of what you found. Do not classify the email — only report and interpret the technical evidence.
"""

    result = llm.invoke(prompt)
    return {"technical_analysis": result.analysis}

from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_email(state):
    email = state["email"]

    prompt = f"""
You are an expert cybersecurity analyst.

Your task is to analyse the email below for phishing indicators. Work through each category in turn before forming your overall assessment. For each category, note what you observe — including when no indicator is present, as absence of indicators is meaningful evidence of legitimacy.

Examine the email systematically across these categories:

1. URGENCY & PRESSURE — Does the email create artificial time pressure, warnings of account suspension, or consequences for inaction?
2. CREDENTIAL & DATA THEFT — Does it request login credentials, passwords, personal information, or financial details?
3. SENDER SPOOFING — Are there signs the sender identity is forged, mismatched, or impersonating a known organisation?
4. SUSPICIOUS LINKS — Are URLs present? Do any use IP addresses as hostnames, URL shorteners, lookalike domain names, or suspicious path keywords (login, verify, account, secure, update)?
5. SOCIAL ENGINEERING — Does the email exploit trust, fear, authority, urgency, or reciprocity to manipulate the recipient into taking action?
6. FINANCIAL PRESSURE — Does it involve unexpected payments, prize notifications, fines, or financial threats?
7. IMPERSONATION — Does it impersonate a bank, government agency, well-known company, or internal colleague?

Email:
{email}

Provide a concise but thorough analysis covering all seven categories. State clearly what evidence you found for each — or that nothing suspicious was observed.
"""

    result = llm.invoke(prompt)
    return {
        "analysis": result.analysis,
        "analysis_leaning": result.leaning,
    }
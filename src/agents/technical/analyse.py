from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_technical(state):
    email = state["email"]

    prompt = f"""
You are a technical cybersecurity analyst specialising in email infrastructure threats, with deep expertise in URL analysis, domain spoofing detection, and malicious payload identification.

Your task is to examine the email below for technical indicators of phishing. Work through each category in turn. For every category, state what you found — or explicitly note that nothing suspicious was observed. Absence of technical indicators is meaningful evidence.

Examine each category systematically:

1. URLs & LINKS — Identify all URLs present. For each, assess:
   - Is the hostname an IP address rather than a domain? (strong indicator)
   - Is a URL shortener used, obscuring the real destination? (strong indicator)
   - Does the path or subdomain contain suspicious keywords: login, verify, account, secure, update, confirm, password, signin? (moderate indicator)
   - Is the TLD unusual or associated with abuse: .xyz, .tk, .ml, .ga, .cf, .click, .top? (moderate indicator)
   - Is the subdomain chain unusually deep (4+ levels)? (moderate indicator)

2. DOMAIN SPOOFING — Look for lookalike domains that impersonate legitimate brands using:
   - Character substitution (paypa1.com, g00gle.com, rnicrosft.com)
   - Added words (amazon-secure.com, paypal-login.net)
   - Different TLD on a known brand (apple.co vs apple.com)

3. SENDER & REPLY-TO — Are From and Reply-To addresses both present? Do they match? Does the sender domain correspond to the claimed organisation?

4. FILE ATTACHMENTS — Are there references to executable or high-risk file types: .exe, .zip, .js, .vbs, .bat, .cmd, .ps1, .jar, .docm, .xlsm? Are there calls to action to open, download, or run a file?

5. EMBEDDED CONTENT — Is there HTML markup, base64-encoded content, or script references embedded in what should be a plain-text email?

Email:
{email}

Provide a concise technical analysis covering all five categories. Do not classify the email — only report and interpret the technical evidence found.

Based on your analysis, provide your overall leaning: "phishing" if you found meaningful technical indicators suggesting deceptive infrastructure, "legitimate" if the email shows no suspicious technical features, or "uncertain" if the evidence is mixed or ambiguous.
"""

    result = llm.invoke(prompt)
    return {
        "technical_analysis": result.analysis,
        "technical_leaning": result.leaning,
    }

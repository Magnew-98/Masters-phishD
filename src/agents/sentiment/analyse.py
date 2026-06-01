from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_sentiment(state):
    email = state["email"]

    prompt = f"""
You are an expert in social engineering and the psychology of deception, specialising in how phishing emails exploit human cognitive biases and emotional responses to manipulate recipients into taking harmful actions.

Your task is to analyse the email below for affective and psychological manipulation indicators. Work through each category systematically. For each, note what you observe — including when no indicator is present, as absence of manipulation is meaningful evidence of legitimacy.

Examine the email across these psychological manipulation categories:

1. URGENCY & TIME PRESSURE — Does the email create artificial deadlines or consequences for inaction? Look for: "act immediately", "expires in 24 hours", "your account will be suspended", "respond today". Distinguish between genuine business urgency (a meeting reminder) and manufactured urgency designed to prevent careful thinking.

2. FEAR & THREAT — Does the email invoke fear to provoke a reactive response? Look for: threats of account closure, legal action, security breaches, financial loss, or negative consequences if the recipient does not comply.

3. AUTHORITY INVOCATION — Does the email exploit deference to authority figures or trusted institutions? Look for: impersonation of banks, government agencies, IT departments, executives (CEO/CFO fraud), or well-known companies. Does it use official-sounding language, titles, or logos to appear legitimate?

4. REWARD & GREED — Does the email use the prospect of unexpected gain to motivate action? Look for: prize notifications, lottery winnings, unclaimed funds, exclusive offers, or financial incentives that require the recipient to take action to claim.

5. SOCIAL PROOF & CONFORMITY — Does the email suggest that others have already complied, or use peer pressure to normalise the requested action? Look for: "all employees have updated their details", "your colleagues have verified", or similar appeals to group behaviour.

6. TRUST & FAMILIARITY EXPLOITATION — Does the email use personal details, casual familiarity, or references to shared context to create a false sense of trust? Look for: use of the recipient's name, references to prior interactions, or impersonation of known contacts.

7. RECIPROCITY — Does the email frame the request as a favour being returned, or suggest the recipient owes a response? Look for: "we have done X for you, now we need you to Y" patterns.

Email:
{email}

Provide a concise analysis covering all seven categories. State clearly what manipulative tactics you found — or explicitly note that nothing suspicious was observed in each category. Do not classify the email — only analyse the psychological and affective evidence.
"""

    result = llm.invoke(prompt)
    return {"sentiment_analysis": result.analysis}

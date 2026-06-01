from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_sentiment(state):
    email = state["email"]

    prompt = f"""
You are a social engineering specialist with 12 years of experience identifying phishing campaigns that exploit psychological vulnerabilities.

Your task is to analyse the email below for psychological manipulation. Focus only on indicators that are genuinely distinctive of phishing — not features that are common in normal business communication.

Key distinction: urgency, authority, deadlines, and consequences are routine in corporate email and are NOT reliable phishing indicators on their own. You are looking for manipulation that would be implausible or out of place in a legitimate professional context.

Examine only these three categories — they are the most distinctive for phishing:

1. REWARD & UNSOLICITED GAIN — Does the email offer unexpected money, prizes, lottery winnings, unclaimed funds, or implausible financial benefits that require personal action to claim? This is rare in legitimate business email and a strong phishing signal when present.

2. IMPLAUSIBLE EXTERNAL AUTHORITY — Does the email claim authority from an organisation that would have no legitimate reason to contact this recipient in this way? Examples: a bank, government body, or tech company contacting a corporate employee with an account verification request unrelated to their work. Distinguish from legitimate internal authority (a manager, IT department, HR).

3. EXTREME ARTIFICIAL URGENCY — Is there a specific, implausible countdown or ultimatum that serves no legitimate business purpose? Examples: "your account will be permanently deleted in 2 hours", "respond within the next 30 minutes or face legal action". Distinguish from normal business deadlines ("please respond by end of week").

For each category, state clearly: not present, present but explainable by legitimate business context, or present in a form that suggests deliberate phishing manipulation.

Email:
{email}

Provide a concise analysis of the three categories. Do not classify the email — only report the affective evidence.
"""

    result = llm.invoke(prompt)
    return {"sentiment_analysis": result.analysis}

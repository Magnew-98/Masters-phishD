from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_sentiment(state):
    email = state["email"]

    prompt = f"""
You are a cognitive security researcher and social engineering specialist with 12 years of experience studying how phishing campaigns exploit human psychological vulnerabilities. You understand both the science of persuasion and the norms of professional business communication.

Your task is to analyse the email below for psychological manipulation indicators. Reason through each category in turn before forming your overall assessment. For each category, explicitly state your finding — including when nothing suspicious is present, as the absence of manipulation is meaningful evidence of legitimacy.

FRAMING CONTEXT — read this before analysing:
Professional business email routinely contains urgency, authority, deadlines, and time pressure as normal features of corporate life. These features alone are NOT phishing indicators. Your task is to identify psychological manipulation that is DISPROPORTIONATE to any plausible business context, or that appears engineered to bypass rational decision-making rather than to communicate a genuine need. For each category, ask: could a legitimate business sender have written this, or does it only make sense as an attempt to manipulate?

Examine the email across these categories:

1. URGENCY & TIME PRESSURE — Is there artificial urgency designed to prevent careful thinking? Normal: "please respond by Friday", "deadline is end of month". Suspicious: "your account will be deleted in 2 hours", "you must act NOW or face consequences", countdown pressure with no legitimate business basis.

2. FEAR & THREAT — Is fear being weaponised beyond normal business stakes? Normal: "failure to comply may result in delays". Suspicious: threats of account closure, legal action, financial penalty, or security breach that appear designed to cause panic rather than inform.

3. AUTHORITY INVOCATION — Is authority being impersonated rather than legitimately claimed? Normal: a manager directing their team, an IT department sending a policy update. Suspicious: implausible authority claims, impersonation of banks, government agencies, or senior executives making unusual requests outside normal channels.

4. REWARD & GREED — Does the email offer unexpected, unsolicited gain? Normal: legitimate bonus or commission notifications. Suspicious: prize notifications, lottery winnings, unclaimed funds, or offers that require personal action to claim an implausible reward.

5. SOCIAL PROOF & CONFORMITY — Is peer pressure being manufactured to suppress scepticism? Suspicious: "all your colleagues have already verified their accounts", "everyone has updated their details — you are the only one remaining".

6. TRUST EXPLOITATION — Is familiarity being fabricated to lower guard? Normal: addressing colleagues by name. Suspicious: false claims of prior interaction, impersonating a known contact, or manufactured intimacy with a stranger.

7. RECIPROCITY — Is obligation being manufactured to extract compliance? Suspicious: framing a request as returning a favour when no genuine prior relationship exists.

Email:
{email}

Provide a concise analysis covering all seven categories. For each, clearly state whether the indicator is absent, present but consistent with legitimate business communication, or present in a form that suggests deliberate psychological manipulation. Do not classify the email — only report the affective evidence.
"""

    result = llm.invoke(prompt)
    return {"sentiment_analysis": result.analysis}

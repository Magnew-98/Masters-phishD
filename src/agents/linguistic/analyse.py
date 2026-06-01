from src.agents.shared_llm import get_llm
from src.schemas.outputs import AnalysisOutput

llm = get_llm().with_structured_output(AnalysisOutput)


def analyse_linguistic(state):
    email = state["email"]

    prompt = f"""
You are a forensic linguist and computational text analyst with 12 years of experience identifying deceptive text in phishing emails. You specialise in surface-level textual anomalies that reveal deliberate manipulation or non-authentic authorship.

Your task is to analyse the email below for linguistic and typographic indicators of phishing. Work through each category systematically. For each, state clearly what you found — or that nothing suspicious was observed. Absence of anomalies is meaningful evidence of authenticity.

CALIBRATION: Occasional typos, informal language, and abbreviations are normal in business email. You are looking for patterns — systematic or deliberate anomalies — not isolated mistakes. A single spelling error in a long email is unremarkable. Systematic misspellings of brand names, or homoglyph substitutions, are significant.

Examine the email across these categories:

1. HOMOGLYPH SUBSTITUTIONS — Are any characters replaced with visually similar alternatives to deceive a reader while evading text filters? Common substitutions:
   - Letters replaced by numbers: a→4, e→3, i→1, o→0, s→5 (e.g. "paypa1", "g00gle", "m1crosoft")
   - Letter pairs replaced by similar letters: rn→m, vv→w, cl→d (e.g. "rnicrosft" for "microsoft")
   - Unicode lookalike characters substituted for standard ASCII
   Look at brand names, domain-like strings, and company names carefully.

2. BRAND NAME AND DOMAIN MISSPELLINGS — Are known brand names spelled with deliberate slight variations designed to impersonate while avoiding exact-match detection? Examples: "Arnazon", "Paypa1", "Micosoft", "Gooogle", "App1e", "Netfliix". Distinguish from genuine typos by considering whether the misspelling is in a brand name specifically.

3. SPELLING AND GRAMMAR PATTERNS — Are there systematic errors suggesting the email was written by a non-native English speaker, auto-generated, or hastily composed to imitate legitimate email? Look for: consistent article errors ("a email", "the informations"), incorrect verb agreement, unusual preposition choices, or unnatural phrasing that would not appear in genuine professional communication. Note: occasional typos are normal — patterns of grammatical error are suspicious.

4. REGISTER INCONSISTENCY — Does the email mix inappropriately formal and informal language, or use stilted phrasing that feels copied from a template rather than written naturally? Phishing emails often use phrases like "Dear Valued Customer", "Kindly revert", "Do the needful", or overly formal constructions that no native speaker would use in the claimed context. Contrast: a colleague emailing informally is legitimate; an email claiming to be from a bank using broken formal English is suspicious.

5. UNUSUAL CAPITALISATION AND PUNCTUATION — Are there irregular patterns beyond normal emphasis? Look for: random capitalisation of common words, EXCESSIVE ALL CAPS for urgency, multiple exclamation marks (!!!), or punctuation patterns inconsistent with the claimed sender's professional context.

Email:
{email}

Provide a concise analysis covering all five categories. For each, state whether no anomaly was found, a minor anomaly consistent with normal email, or a meaningful anomaly suggesting deliberate manipulation. Do not classify the email — only report the linguistic evidence.

Based on your analysis, provide your overall leaning: "phishing" if you found meaningful linguistic anomalies suggesting deceptive or non-authentic authorship, "legitimate" if the text is consistent with genuine business communication, or "uncertain" if the evidence is mixed.
"""

    result = llm.invoke(prompt)
    return {
        "linguistic_analysis": result.analysis,
        "linguistic_leaning": result.leaning,
    }

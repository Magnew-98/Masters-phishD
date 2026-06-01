from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def classify_linguistic(state):
    linguistic_analysis = state["linguistic_analysis"]

    prompt = f"""
You are a cybersecurity classification expert specialising in linguistic forensics, with experience identifying phishing emails through textual and typographic analysis.

Your task is to classify the email as phishing or legitimate based solely on the linguistic evidence in the analysis below. Reason through the evidence step by step before concluding.

Classification definitions:
- LEGITIMATE: Text consistent with genuine business or personal communication. May contain typos or informal language but lacks systematic anomalies or deliberate brand impersonation.
- PHISHING: Text showing deliberate linguistic manipulation — homoglyph substitutions in brand names, systematic grammatical patterns inconsistent with the claimed sender, or register anomalies that reveal non-authentic authorship.

Decision process:
1. Were meaningful linguistic anomalies identified — not just minor typos, but patterns suggesting deliberate manipulation or non-authentic authorship?
2. Are any brand names or domain-like strings misspelled in a way that suggests impersonation rather than a genuine error?
3. Do grammar or register findings indicate the email was written by someone other than who they claim to be?
4. Step back: could a legitimate business sender have written this email with this specific pattern of language? Or does the linguistic evidence only make sense if the sender is concealing their identity?

Calibration guidance:
- No anomalies or only minor isolated typos → legitimate, high confidence (0.85+)
- Clear homoglyph substitution in a brand name → phishing, high confidence (0.85+)
- Systematic grammar or register anomalies without brand misspelling → phishing, moderate confidence (0.65–0.80)
- Single suspicious pattern without corroboration → lean phishing, low-moderate confidence (0.60–0.70)
- Ambiguous evidence → uncertain, lean legitimate (0.50–0.60)
- Confidence: 0.5 = genuinely uncertain, 0.7 = moderate evidence, 0.9+ = clear deceptive manipulation

Linguistic analysis:
{linguistic_analysis}

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

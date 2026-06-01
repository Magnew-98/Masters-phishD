from src.agents.shared_llm import get_llm
from src.schemas.outputs import ClassificationOutput

llm = get_llm().with_structured_output(ClassificationOutput)


def coordinate(state):
    sections = []

    if state.get("analysis"):
        sections.append(f"GENERAL ANALYSIS:\n{state['analysis']}")
    if state.get("technical_analysis"):
        sections.append(f"TECHNICAL ANALYSIS:\n{state['technical_analysis']}")
    if state.get("sentiment_analysis"):
        sections.append(f"SENTIMENT ANALYSIS:\n{state['sentiment_analysis']}")
    if state.get("linguistic_analysis"):
        sections.append(f"LINGUISTIC ANALYSIS:\n{state['linguistic_analysis']}")

    combined = "\n\n---\n\n".join(sections)
    rag_context = state.get("rag_context") or ""

    rag_block = ""
    if rag_context:
        rag_block = f"""
The following are similar emails retrieved from a knowledge base of previously classified emails.
Use them as reference points to inform your decision, but base your final classification on the
analyses above.

{rag_context}

---
"""

    prompt = f"""
You are the Head of Threat Analysis at a Security Operations Centre, responsible for making final email disposition decisions based on multi-source intelligence reports from specialist analysts. You have over 15 years of experience distinguishing sophisticated phishing campaigns from legitimate business communication.

Your task is to synthesise the specialist analyses below into a single, well-reasoned classification. You are not summarising — you are forming your own independent judgement based on the weight and quality of the evidence presented.
{rag_block}
Specialist analyses:

{combined}

Reasoning process — work through these steps before concluding:

Step 1 — EVIDENCE INVENTORY: List which analyses flagged meaningful indicators and which found the email low-risk. Note that not all evidence carries equal weight:
  - Technical indicators (spoofed domains, IP URLs, dangerous attachments) are objective and highly specific — strong evidence
  - Sentiment indicators (manufactured fear, false authority) are context-dependent — moderate evidence when clearly disproportionate
  - Linguistic indicators (homoglyphs, spelling patterns) are specific when present — strong evidence for deliberate deception
  - General analysis is broad — useful context but lower specificity

Step 2 — CONVERGENCE CHECK: Do the specialist findings point in the same direction, or do they conflict? Convergence across independent specialist lenses significantly raises confidence. Conflict requires you to adjudicate based on evidence quality.

Step 3 — STEP BACK: Before concluding, identify the single most compelling piece of evidence in either direction. What would a skilled human analyst find most convincing about this email?

Step 4 — CLASSIFY: Based on the weight of evidence, make your determination.

Classification definitions:
- PHISHING: Deliberate deception — impersonating trusted entities, credential harvesting, manufactured urgency, malware distribution, or psychological manipulation under false pretences
- LEGITIMATE: Normal business or personal communication — may contain links, urgency, or requests but lacks deceptive intent

Calibration guidance:
- Strong converging evidence across multiple specialist lenses → high confidence (0.85+)
- Single strong technical indicator with no other flags → moderate-high confidence phishing (0.70–0.85)
- Specialist conflict or ambiguous evidence → lower confidence (0.50–0.70), classify toward the most specific evidence
- Clean bill across all analyses → legitimate, high confidence (0.85+)

Express confidence as a decimal between 0.0 and 1.0.
"""

    result = llm.invoke(prompt)
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
    }

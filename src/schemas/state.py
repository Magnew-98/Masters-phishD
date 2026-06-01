from typing import TypedDict, Optional

class EmailState(TypedDict):
    email: str

    # binary agent
    analysis: Optional[str]

    # rag context
    rag_context: Optional[str]

    # specialist agents
    technical_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    linguistic_analysis: Optional[str]

    # final outputs
    prediction: Optional[str]
    confidence: Optional[float]
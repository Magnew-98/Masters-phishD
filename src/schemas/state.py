from typing import TypedDict, Optional

class EmailState(TypedDict):
    email: str

    # binary agent
    analysis: Optional[str]

    # rag context
    rag_context: Optional[str]

    # specialist agents — analysis text
    technical_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    linguistic_analysis: Optional[str]

    # specialist agents — directional leanings
    analysis_leaning: Optional[str]
    technical_leaning: Optional[str]
    sentiment_leaning: Optional[str]
    linguistic_leaning: Optional[str]

    # final outputs
    prediction: Optional[str]
    confidence: Optional[float]
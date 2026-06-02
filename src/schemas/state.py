from typing import TypedDict, Optional

class EmailState(TypedDict):
    email: str
    rag_context: Optional[str]

    analysis: Optional[str]
    technical_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    linguistic_analysis: Optional[str]

    analysis_leaning: Optional[str]
    technical_leaning: Optional[str]
    sentiment_leaning: Optional[str]
    linguistic_leaning: Optional[str]

    prediction: Optional[str]
    confidence: Optional[float]

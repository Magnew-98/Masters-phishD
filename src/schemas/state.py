from typing import TypedDict, Optional

class EmailState(TypedDict):
    email: str

    # intermediate reasoning
    analysis: Optional[str]

    # final outputs
    prediction: Optional[str]
    confidence: Optional[float]
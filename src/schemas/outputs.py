from pydantic import BaseModel
from typing import Literal

class AnalysisOutput(BaseModel):
    analysis: str


class ClassificationOutput(BaseModel):
    prediction: Literal["phishing", "legitimate"]
    confidence: float
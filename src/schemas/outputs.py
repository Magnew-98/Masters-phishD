from pydantic import BaseModel, field_validator
from typing import Literal


class AnalysisOutput(BaseModel):
    analysis: str
    leaning: Literal["phishing", "legitimate", "uncertain"]


class ClassificationOutput(BaseModel):
    prediction: Literal["phishing", "legitimate"]
    confidence: float

    @field_validator("confidence")
    @classmethod
    def fix_confidence(cls, v: float) -> float:
        if v > 1.0:
            return v / 100.0
        return v

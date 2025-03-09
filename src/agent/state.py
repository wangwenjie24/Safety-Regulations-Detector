"""Define the state structures for the agent."""

from __future__ import annotations

import operator
from typing import List
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, Field

class Rule(BaseModel):
    """Safety regulation rule."""
    item: str = Field(None, description="Item of the rule")

class Rules(BaseModel):
    """List of rules."""
    rules: List[Rule] = Field(
        description="List of rules.",
    )

class Detection(BaseModel):
    """Detection result."""
    conclusion: bool = Field(
        description="Conclusion based on the assessment.",
    )
    explanation: str = Field(
        description="Explanation supporting the conclusion.",
    )
    targets: list[str] = Field(
        description="Descriptions of violating individuals for annotation.",
    )

class DetctionResult(BaseModel):
    conclusion: bool = Field(
        description="Conclusion based on the assessment.",
    )
    explanation: str = Field(
        description="Explanation supporting the conclusion.",
    )
    targets: list[str] = Field(
        description="Descriptions of violating individuals for annotation.",
    )
    coordinates: list[Coordinates] = Field(
        description="Bounding box of the detection.",
    )

class Coordinates(BaseModel):
    coordinates: List[int] = Field(
        description="Detection bounding box coordinates.",
    )

class AnalysisState(TypedDict):
    safety_items: str
    image_url: str
    rules: list[Rule]
    completed_detection_results: Annotated[list, operator.add]
    explanation_list: list[str]
    coordinates_list: list[Coordinates]
    conclusion: bool    
    marked_image: str
    final_result: str

class AnalysisInputState(TypedDict):
    safety_items: str
    image_url: str

class AnalysisOutputState(TypedDict):
    marked_image: str
    final_result: str

class DetectionState(TypedDict):
    rule: str
    image_url: str
    detection: Detection
    targets: list[str]
    target: str
    coordinates: Annotated[list[Coordinates], operator.add]
    completed_detection_results: list[DetctionResult]

class DetectionOutputState(TypedDict):
    completed_detection_results: list[DetctionResult]

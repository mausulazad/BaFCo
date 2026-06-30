from typing import Annotated, List
from pydantic import BaseModel, Field, RootModel, ConfigDict, field_validator, ValidationInfo


class LayoutElement(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Unique ID for this detected region.")
    reasoning: str = Field(..., description="Brief one-line reasoning for classification.")
    class_: str = Field(
        ...,
        alias="class",
        description="Category name (matches one of the form field categories)."
    )
    bbox: Annotated[List[float], Field(
        min_length=4, max_length=4,
        description="[X, Y, W, H] bounding box with X,Y = top-left origin."
    )]
    confidence: Annotated[float, Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )]

    @field_validator("class_", mode="before")
    @classmethod
    def check_category(cls, v, info: ValidationInfo):
        categories = info.context.get("categories") if info.context else None
        if categories is None:
            return v
        s = str(v).strip()
        if s.lower() not in categories:
            raise ValueError(f"invalid class: {s}")
        return s

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v):
        try:
            v = list(v)
        except (TypeError, ValueError):
            raise ValueError("bbox must be a list")
        if len(v) != 4:
            raise ValueError("bbox must have 4 numbers [x, y, w, h]")
        try:
            v = [float(x) for x in v]
        except (TypeError, ValueError):
            raise ValueError("bbox values must be numbers or numeric strings")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def check_confidence(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number or numeric string")
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in the range [0.0, 1.0]")
        return v


class LayoutAnalysisOutput(RootModel[List[LayoutElement]]):
    """Top-level output: list of layout elements."""
    pass

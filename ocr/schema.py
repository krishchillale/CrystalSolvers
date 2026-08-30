"""
Data contract between the OCR module and the entity-extraction module.

Keep this file the single source of truth for the handoff shape — if it
changes, both halves of the OCR pipeline need to agree.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OCREngine(str, Enum):
    qwen3_vl = "qwen3-vl-8b"
    qwen2_5_vl_3b = "qwen2.5-vl-3b"
    tesseract = "tesseract"


class BoundingBox(BaseModel):
    """Approximate region of the page this text block came from.

    Coordinates are normalized 0–1 (fraction of page width/height), so they
    stay valid regardless of the image resolution used at OCR time.
    """

    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)


class TextBlock(BaseModel):
    text: str
    bbox: Optional[BoundingBox] = None
    confidence: float = Field(..., ge=0, le=1)
    engine: OCREngine
    is_handwritten: Optional[bool] = None  # set when the engine can flag it


class OCRResult(BaseModel):
    document_id: str
    page_number: int = 1
    full_text: str
    blocks: list[TextBlock]
    engine_used: OCREngine
    preprocessing_applied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

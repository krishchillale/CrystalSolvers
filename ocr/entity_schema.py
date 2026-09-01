"""
Schema for structured field extraction — the "what did this document
actually say, clinically" layer, on top of raw OCR text.

Deliberately excludes administrative/contact fields (hospital name,
doctor name, phone numbers, emails, addresses) — those are on the
document and still readable in the raw OCR full_text if ever needed,
but are never pulled into the structured extraction or CSV/JSON output.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Investigation(BaseModel):
    name: str
    result: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None  # e.g. "High", "Low", "Normal", null if unclear


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None


class DocumentExtraction(BaseModel):
    document_type: Optional[str] = None  # e.g. "lab_report", "prescription"
    patient_name: Optional[str] = None
    age_sex: Optional[str] = None
    report_date: Optional[str] = None
    investigations: list[Investigation] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
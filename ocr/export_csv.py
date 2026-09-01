"""
Converts a DocumentExtraction into a single flat CSV, named after the
input file (e.g. test1jpeg.jpeg -> test1jpeg.csv). Patient/report-level
fields are repeated on every row so each row is self-contained and the
file opens cleanly in Excel with no separate lookup needed.

Administrative fields (hospital, doctor, contact info) are intentionally
absent — see entity_schema.py.

If a page has no investigations or medications, one row is still written
so the document-level info isn't lost.
"""

from __future__ import annotations

import csv

from entity_schema import DocumentExtraction

CSV_FIELDS = [
    "document_id",
    "page_number",
    "document_type",
    "patient_name",
    "age_sex",
    "report_date",
    "category",  # "investigation" or "medication"
    "name",
    "result_or_dose",
    "unit_or_frequency",
    "reference_range_or_duration",
    "flag",
]


def _base_row(document_id: str, page_number: int, extraction: DocumentExtraction) -> dict:
    return {
        "document_id": document_id,
        "page_number": page_number,
        "document_type": extraction.document_type,
        "patient_name": extraction.patient_name,
        "age_sex": extraction.age_sex,
        "report_date": extraction.report_date,
    }


def extraction_to_rows(
    document_id: str, page_number: int, extraction: DocumentExtraction
) -> list[dict]:
    base = _base_row(document_id, page_number, extraction)
    rows = []

    for inv in extraction.investigations:
        rows.append(
            {
                **base,
                "category": "investigation",
                "name": inv.name,
                "result_or_dose": inv.result,
                "unit_or_frequency": inv.unit,
                "reference_range_or_duration": inv.reference_range,
                "flag": inv.flag,
            }
        )
    for med in extraction.medications:
        rows.append(
            {
                **base,
                "category": "medication",
                "name": med.name,
                "result_or_dose": med.dose,
                "unit_or_frequency": med.frequency,
                "reference_range_or_duration": med.duration,
                "flag": None,
            }
        )

    if not rows:
        # Still record the document-level fields even with no line items.
        rows.append({**base, "category": None, "name": None, "result_or_dose": None,
                      "unit_or_frequency": None, "reference_range_or_duration": None,
                      "flag": None})

    return rows


def write_csv(output_path: str, all_rows: list[dict]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
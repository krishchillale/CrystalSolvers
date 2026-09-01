"""
CLI entrypoint for running the preprocess -> OCR -> structured extraction
pipeline on a single file. Handles both plain images (jpg/png) and PDFs
(each page processed separately).

    python main.py path/to/document.jpg
    python main.py path/to/document.pdf

Output: for an input named "test1.jpeg", writes:
    test1.json  — combined raw OCR + structured extraction, per page
    test1.csv   — flat table, one row per investigation/medication
                  (patient/report fields repeated on every row)

Both are named after the input file and OVERWRITTEN on each run for that
same input — this is per-document output, not an accumulating log.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ocr_engine import run_ocr
from export_csv import extraction_to_rows, write_csv
from extract_structured import extract_structured
from pdf_utils import is_pdf, pdf_to_images
from preprocess import preprocess_image, preprocess_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _get_page_images(path: str):
    """Yields (processed_image, preprocessing_steps) for each page."""
    if is_pdf(path):
        for img in pdf_to_images(path):
            yield preprocess_image(img)
    else:
        yield preprocess_pipeline(path)


def process_file(path: str) -> list[dict]:
    """Returns a list of per-page dicts: {"ocr": OCRResult, "extraction":
    DocumentExtraction or None}."""
    pages = []
    for page_num, (processed, steps) in enumerate(_get_page_images(path), start=1):
        result = run_ocr(processed)
        result.page_number = page_num
        result.preprocessing_applied = steps

        extraction = None
        if result.engine_used.value.startswith("qwen"):
            tier = "8b" if result.engine_used.value == "qwen3-vl-8b" else "3b"
            extraction = extract_structured(processed, tier=tier)
            if extraction is None:
                print(f"[page {page_num}] Structured extraction failed — "
                      f"see log above. Raw OCR text still saved.")

        pages.append({"ocr": result, "extraction": extraction})

    return pages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_or_pdf_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    stem = Path(input_path).stem  # e.g. "test1jpeg" from "test1jpeg.jpeg"

    pages = process_file(input_path)

    # --- JSON: combined OCR + structured extraction, per page ---
    json_output = [
        {
            "ocr": page["ocr"].model_dump(),
            "extraction": page["extraction"].model_dump() if page["extraction"] else None,
        }
        for page in pages
    ]
    json_path = f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")

    # --- CSV: flat entity table, only for pages that had a successful
    # structured extraction ---
    all_rows = []
    for page in pages:
        if page["extraction"] is not None:
            all_rows.extend(
                extraction_to_rows(page["ocr"].document_id, page["ocr"].page_number, page["extraction"])
            )

    if all_rows:
        csv_path = f"{stem}.csv"
        write_csv(csv_path, all_rows)
        print(f"Saved {csv_path} ({len(all_rows)} rows)")
    else:
        print("No successful structured extractions — CSV not written.")
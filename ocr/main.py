"""
CLI entrypoint for running the preprocess -> OCR pipeline on a single file.
Handles both plain images (jpg/png) and PDFs (each page processed
separately). Useful while developing before the FastAPI server is wired up.

    python main.py path/to/document.jpg
    python main.py path/to/document.pdf
"""

from __future__ import annotations

import json
import sys

from ocr_engine import run_ocr
from pdf_utils import is_pdf, pdf_to_images
from preprocess import preprocess_image, preprocess_pipeline


def process_file(path: str) -> list:
    """Returns a list of OCRResult -- one per page. Plain images are
    treated as a single-page document."""
    if is_pdf(path):
        page_images = pdf_to_images(path)
        results = []
        for page_num, img in enumerate(page_images, start=1):
            processed, steps = preprocess_image(img)
            result = run_ocr(processed)
            result.page_number = page_num
            result.preprocessing_applied = steps
            results.append(result)
        return results

    processed, steps = preprocess_pipeline(path)
    result = run_ocr(processed)
    result.preprocessing_applied = steps
    return [result]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_or_pdf_path>")
        sys.exit(1)

    results = process_file(sys.argv[1])
    print(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False))

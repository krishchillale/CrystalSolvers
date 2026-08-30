"""
FastAPI wrapper around the OCR pipeline. Run standalone during development:

    uvicorn app.server:app --reload --port 8001

Once Member 4's main backend is up, this can either stay a separate
service (call it over HTTP) or its router can be mounted directly into
the main FastAPI app — either way the request/response shape is the same.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from ocr_engine import run_ocr
from pdf_utils import is_pdf, pdf_to_images
from preprocess import preprocess_image, preprocess_pipeline

app = FastAPI(title="OCR / Document AI Service")

ALLOWED_TYPES = ("image/jpeg", "image/png", "image/jpg", "application/pdf")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/extract")
async def extract(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. "
            f"Allowed: {ALLOWED_TYPES}",
        )

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename).suffix, delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        tmp_path = tmp.name

    try:
        if is_pdf(file.filename):
            results = []
            for page_num, img in enumerate(pdf_to_images(tmp_path), start=1):
                processed, steps = preprocess_image(img)
                result = run_ocr(processed)
                result.page_number = page_num
                result.preprocessing_applied = steps
                results.append(result)
            return results

        img, steps = preprocess_pipeline(tmp_path)
        result = run_ocr(img)
        result.preprocessing_applied = steps
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)

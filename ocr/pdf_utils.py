"""
Converts PDF pages to images so the existing preprocess -> OCR pipeline
(which is image-only) can handle PDFs too, without needing any change to
preprocess.py or ocr_engine.py.

Uses PyMuPDF (fitz) rather than pdf2image, because pdf2image needs the
external `poppler` binary installed separately (extra Windows setup pain),
while PyMuPDF is a pure pip install.
"""

from __future__ import annotations

import pymupdf as fitz  # `fitz` is the old import name, being deprecated
import numpy as np


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Render every page of a PDF to a BGR image array (same format
    cv2.imread would give you), so it can go straight into preprocess.py.

    dpi=300 is a reasonable default for OCR — higher improves small-text
    accuracy at the cost of processing time; don't go below ~200.
    """
    zoom = dpi / 72  # PDF points are 72 per inch
    matrix = fitz.Matrix(zoom, zoom)

    images = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # PyMuPDF gives RGB; cv2 elsewhere in the pipeline expects BGR.
            img_bgr = img[:, :, ::-1].copy()
            images.append(img_bgr)
    finally:
        doc.close()

    return images


def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

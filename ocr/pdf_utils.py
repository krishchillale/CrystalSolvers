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


def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[np.ndarray]:
    """Render every page of a PDF to a BGR image array (same format
    cv2.imread would give you), so it can go straight into preprocess.py.

    dpi=150 balances OCR accuracy against VLM memory use — 300 DPI renders
    are roughly 4x the pixel count (and thus far more vision-encoder
    tokens/VRAM), which is what caused CUDA OOM on a 6GB card even at
    the 3B tier. Go higher only if small text is getting missed and your
    GPU has headroom to spare.
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

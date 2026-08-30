"""
Preprocessing: get a messy phone photo / scan into the best possible shape
before OCR ever sees it. Each step is independent so you can turn steps
on/off while tuning against real sample documents.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray) -> np.ndarray:
    """Light denoising — fastNlMeans is a good default for phone-camera noise."""
    return cv2.fastNlMeansDenoising(img, h=10)


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate and correct rotation using the minAreaRect of text pixels.

    Returns the corrected image and the detected skew angle (useful to log —
    large angles on a supposedly-flat scan often indicate a bad crop).
    """
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """Improves contrast for uneven lighting (common in phone photos of paper)."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def preprocess_image(
    img: np.ndarray, apply_threshold: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Runs the standard pipeline on an already-loaded image array (e.g. a
    PDF page rendered to an array) and returns the processed image plus a
    log of which steps ran (so it can go into OCRResult.preprocessing_applied).
    """
    steps_applied = []

    gray = to_grayscale(img)
    steps_applied.append("grayscale")

    denoised = denoise(gray)
    steps_applied.append("denoise")

    deskewed, angle = deskew(denoised)
    steps_applied.append(f"deskew(angle={angle:.1f})")

    if apply_threshold:
        deskewed = adaptive_threshold(deskewed)
        steps_applied.append("adaptive_threshold")

    return deskewed, steps_applied


def preprocess_pipeline(
    path: str, apply_threshold: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Convenience wrapper for the common case: preprocess a single image
    file straight from disk. PDF pages should use preprocess_image()
    directly since they're already loaded as arrays — see main.py.
    """
    img = load_image(path)
    processed, steps = preprocess_image(img, apply_threshold=apply_threshold)
    return processed, ["load"] + steps

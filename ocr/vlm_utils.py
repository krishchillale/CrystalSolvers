"""
Caps image resolution before it reaches the VLM. Vision-language models
tokenize images into patches — a bigger image means more tokens means
more VRAM for activations, independent of the model's own weight size.
This is what caused CUDA OOM on 300 DPI PDF renders even on the smaller
3B tier: the image itself was too large, not the model.

max_side=1280 is a reasonable default: large enough to keep small text
(lab values, medicine names) legible, small enough to stay safely within
a 6GB card's VRAM budget alongside the model weights + KV cache.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_vlm(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img  # already small enough, don't upscale

    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

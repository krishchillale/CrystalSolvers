"""
OCR engine wrapper. The rest of the pipeline only ever calls `run_ocr()` —
everything about *which* model is running is decided in here, so swapping
engines never touches the schema, the API layer, or your partner's code.

Primary: Qwen3-VL-8B, 4-bit quantized (best real-world accuracy on
Hindi/Devanagari text per our benchmark research — see project notes).
On a 6GB card this is genuinely tight (~4.8GB just for INT4 weights + KV
cache at default settings, before the vision encoder's activation memory
on a full document image) — it may not fit, especially on larger scans.

Secondary: Qwen2.5-VL-3B, 4-bit quantized. Much safer on 6GB (~2GB weights),
still meaningfully better than Tesseract on printed/scanned text, though
weaker than the 8B model specifically on real-world Devanagari.

Fallback: Tesseract. Runs on CPU, much weaker on handwriting and Indic
scripts, but always works regardless of GPU/VRAM.

USE_QWEN_TIER controls which VLM tier to try first — start at "8b", and if
you hit repeated CUDA OOM errors on real documents, drop to "3b".
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

import cv2
import numpy as np

from schema import BoundingBox, OCREngine, OCRResult, TextBlock

logger = logging.getLogger(__name__)

# "8b" tries Qwen3-VL-8B (best accuracy, tight on 6GB) first, falling back
# to "3b" automatically on OOM. "3b" skips straight to the smaller model.
# "off" skips VLMs entirely and goes straight to Tesseract.
#
# NOTE: confirmed on an RTX 3050 6GB — Qwen3-VL-8B does NOT fit even at
# 4-bit (accelerate can't auto-offload under bitsandbytes int4 without extra
# config). Default set to "3b" to skip the guaranteed-fail 8b attempt.
USE_QWEN_TIER: Literal["8b", "3b", "off"] = "3b"

# Point this at your actual Tesseract install if it's not on PATH.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_qwen_model = None
_qwen_processor = None
_qwen_model_id = None

_QWEN_MODEL_IDS = {
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def _load_qwen(tier: Literal["8b", "3b"]):
    """Lazy-load, 4-bit quantized via bitsandbytes so it fits on small VRAM.
    Caches the loaded model — if you call this with a *different* tier than
    what's cached, it reloads (e.g. after an OOM fallback from 8b to 3b).
    """
    global _qwen_model, _qwen_processor, _qwen_model_id

    model_id = _QWEN_MODEL_IDS[tier]
    if _qwen_model is not None and _qwen_model_id == model_id:
        return _qwen_model, _qwen_processor

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    import torch

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # extra memory saving, small quality cost
    )

    _qwen_processor = AutoProcessor.from_pretrained(model_id)
    _qwen_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    _qwen_model_id = model_id
    return _qwen_model, _qwen_processor


def get_qwen_for_tier(tier: Literal["8b", "3b"]):
    """Public accessor so other modules (e.g. structured extraction) can
    reuse the already-loaded model instead of loading it a second time.
    """
    return _load_qwen(tier)


def _run_qwen_tier(img: np.ndarray, tier: Literal["8b", "3b"]) -> OCRResult | None:
    try:
        import torch
        model, processor = _load_qwen(tier)
    except Exception as e:  # noqa: BLE001 — deliberately broad: any load failure -> fallback
        logger.warning("Qwen (%s) load failed: %s", tier, e)
        return None

    from PIL import Image
    from vlm_utils import resize_for_vlm

    img = resize_for_vlm(img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    prompt = (
        "Transcribe all text in this medical document image exactly as written, "
        "preserving line breaks. The document may contain Hindi, Marathi, or "
        "English text, and may be printed or handwritten. Do not summarize or "
        "translate — transcribe verbatim."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # apply_chat_template(tokenize=True) does NOT process embedded images —
    # it only builds the text prompt. The image has to be passed separately
    # through the processor call below so it actually reaches the model.
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=[pil_img], return_tensors="pt"
    ).to(model.device)

    try:
        output_ids = model.generate(**inputs, max_new_tokens=2048)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM running Qwen (%s) — falling back.", tier)
        torch.cuda.empty_cache()
        return None

    # Slice off the input prompt tokens so we only decode the newly
    # generated reply, not the whole system+user+assistant transcript.
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = text.strip().strip("`").strip()  # some models wrap output in ``` fences

    del inputs, output_ids
    torch.cuda.empty_cache()  # release activation memory before next page/call

    engine = OCREngine.qwen3_vl if tier == "8b" else OCREngine.qwen2_5_vl_3b

    # NOTE: plain transcription mode doesn't return bounding boxes.
    # If per-field source regions matter for your "evidence" requirement
    # (blueprint section 6), prompt in grounding mode instead, or run a
    # lightweight layout-detection pass separately.
    block = TextBlock(
        text=text.strip(),
        bbox=None,
        confidence=0.9,  # Qwen doesn't emit a native score; placeholder pending calibration
        engine=engine,
    )

    return OCRResult(
        document_id=str(uuid.uuid4()),
        full_text=text.strip(),
        blocks=[block],
        engine_used=engine,
    )


def _run_tesseract(img: np.ndarray) -> OCRResult:
    import pytesseract
    from pytesseract import Output

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # hin+mar+eng: adjust to whichever languages you've actually installed
    # tessdata for (see README).
    data = pytesseract.image_to_data(
        img, lang="hin+mar+eng", output_type=Output.DICT
    )

    blocks = []
    h, w = img.shape[:2]
    lines: dict[int, list[int]] = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        lines.setdefault(data["line_num"][i], []).append(i)

    full_text_parts = []
    for line_idxs in lines.values():
        line_text = " ".join(data["text"][i] for i in line_idxs).strip()
        if not line_text:
            continue
        xs = [data["left"][i] for i in line_idxs]
        ys = [data["top"][i] for i in line_idxs]
        ws = [data["left"][i] + data["width"][i] for i in line_idxs]
        hs = [data["top"][i] + data["height"][i] for i in line_idxs]
        confs = [int(data["conf"][i]) for i in line_idxs if int(data["conf"][i]) >= 0]
        avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0

        bbox = BoundingBox(
            x=min(xs) / w,
            y=min(ys) / h,
            width=(max(ws) - min(xs)) / w,
            height=(max(hs) - min(ys)) / h,
        )
        blocks.append(
            TextBlock(
                text=line_text,
                bbox=bbox,
                confidence=avg_conf,
                engine=OCREngine.tesseract,
            )
        )
        full_text_parts.append(line_text)

    return OCRResult(
        document_id=str(uuid.uuid4()),
        full_text="\n".join(full_text_parts),
        blocks=blocks,
        engine_used=OCREngine.tesseract,
        warnings=(
            ["Tesseract fallback in use — accuracy on handwriting and "
             "Hindi/Marathi text is significantly weaker than a VLM engine. "
             "Check USE_QWEN_TIER if this wasn't intended."]
        ),
    )


def run_ocr(img: np.ndarray) -> OCRResult:
    """Single entry point the rest of the pipeline calls.

    Tries the configured Qwen tier first (with automatic 8b -> 3b fallback
    on OOM), then falls back to Tesseract on any failure (no GPU, model not
    downloaded, both tiers OOM, etc.) so the pipeline never crashes outright
    — it just runs at fallback quality and says so via `OCRResult.warnings`.
    """
    if USE_QWEN_TIER == "8b":
        result = _run_qwen_tier(img, "8b")
        if result is not None:
            return result
        logger.warning("Falling back from 8b to 3b tier.")
        result = _run_qwen_tier(img, "3b")
        if result is not None:
            return result
    elif USE_QWEN_TIER == "3b":
        result = _run_qwen_tier(img, "3b")
        if result is not None:
            return result

    return _run_tesseract(img)
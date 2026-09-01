"""
Structured extraction: instead of parsing raw OCR text with regex, we ask
the same VLM to read the image directly and return a defined JSON shape.
This tends to be more accurate than post-hoc parsing, since the model can
use visual layout (which column is which, what's grouped together) that's
lost once everything's flattened to plain text.

Requires a Qwen tier to already be working (i.e. GPU path) — this doesn't
have a Tesseract-equivalent fallback, since regex-based structuring from
messy Tesseract text is a separate, lower-accuracy approach your partner's
module may still want to build for robustness. This is the fast/local-dev
path.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import cv2
import numpy as np
from pydantic import ValidationError

from entity_schema import DocumentExtraction
from ocr_engine import get_qwen_for_tier

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Read this medical document image and extract the following fields as a \
single JSON object. Use these exact keys:

{
  "document_type": "lab_report" | "prescription" | "discharge_summary" | "other",
  "patient_name": string or null,
  "age_sex": string or null,
  "report_date": string or null,
  "investigations": [
    {"name": string, "result": string or null, "unit": string or null, \
"reference_range": string or null, "flag": string or null}
  ],
  "medications": [
    {"name": string, "dose": string or null, "frequency": string or null, \
"duration": string or null}
  ]
}

CRITICAL rule for investigations — read carefully:
Lab reports often group tests under section headers like "Physical \
Examination", "Chemical Examination", or "Microscopic Examination". These \
section headers are NOT test names and must NOT become an investigation \
entry themselves, and must NOT have any test's result attached to them. \
This applies to EVERY such section on the document, not just some of them \
— check the whole page, including near the top. Instead, create ONE \
SEPARATE investigation entry for EACH individual test result listed under \
each section.

For example, if the document shows:
  Physical Examination
    Colour ...... Dark yellow
    Appearance ... Clear
  Chemical Examination
    pH ......... 5.5 ......... 5-8
    Glucose ..... Neg ......... Negative
you must output FOUR separate entries — "Colour", "Appearance", "pH", and
"Glucose" — each with its own single result. NOT entries named after the
section headers ("Physical Examination", "Chemical Examination") with
multiple results crammed together.

This ALSO applies to the very FIRST test listed under each section header
— a common mistake is to accidentally use the section header's own text as
the "name" for that first test. In the example above, the first entry
must be named "Colour" (not "Physical Examination"), and "pH" must be
named "pH" (not "Chemical Examination"). The section header text itself
must NEVER appear as a "name" value anywhere in the output — check this
specifically for the first item under every section.

CRITICAL rule for unit vs reference_range — these are different columns,
do not confuse them:
- "unit" is the measurement unit the result is expressed in (e.g. mg/dL,
  /hpf., %, mmol/L). Many tests have NO unit (e.g. a pH reading, a
  positive/negative chemical test) — for those, "unit" must be null.
- "reference_range" is the normal/expected range or value for that test
  (often in a column labeled "Biological Ref. Interval", "Reference
  Range", "Normal Range", or similar). This is a DIFFERENT value from
  the unit and must go in "reference_range", never in "unit" — even when
  "unit" would otherwise be null.
- Do not leave "reference_range" empty just because "unit" is empty —
  check the correct column for each.

Other rules:
- Include EVERY individual investigation/medication line on the entire \
document — check top to bottom carefully, don't stop partway through a \
section or skip any line, including ones near section boundaries.
- If the document has no investigations or no medications, use an empty list.
- If a field isn't present in the document, use null — don't guess or \
invent values.
- Do NOT extract or include hospital/clinic name, doctor name, phone \
numbers, email addresses, physical addresses, or any other administrative \
or contact information — even if present on the document. Only clinical/\
patient information belongs in the output.
- Output ONLY the JSON object. No markdown fences, no explanation, no \
extra text before or after.
"""

_COMPLETENESS_CHECK_TEMPLATE = """

For reference, here is a plain-text transcript of this same document from \
a separate OCR pass. Use it to DOUBLE-CHECK you haven't missed any test \
line that appears in the image — if you see a test result in this \
transcript that isn't in your investigations/medications list yet, add \
it. The transcript may have minor OCR errors, so prefer what you read in \
the image itself for the actual values, but use the transcript as a \
checklist to catch anything you skipped:

--- TRANSCRIPT START ---
{ocr_text}
--- TRANSCRIPT END ---
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        text = text.removeprefix("json").strip()
    return text.strip()


def extract_structured(
    img: np.ndarray,
    tier: Literal["8b", "3b"] = "3b",
    ocr_text: str | None = None,
) -> DocumentExtraction | None:
    """Returns a validated DocumentExtraction, or None if extraction/parsing
    failed (check logs — the pipeline should fall back to just using the
    raw OCRResult.full_text in that case, not crash).

    ocr_text: optionally pass the already-extracted raw transcript (from
    run_ocr()) so the model can cross-check completeness against it —
    helps catch lines the structured pass would otherwise silently skip.
    """
    import torch

    try:
        model, processor = get_qwen_for_tier(tier)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load Qwen (%s) for structured extraction: %s", tier, e)
        return None

    from PIL import Image
    from vlm_utils import resize_for_vlm

    img = resize_for_vlm(img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    prompt_text = _EXTRACTION_PROMPT
    if ocr_text:
        prompt_text += _COMPLETENESS_CHECK_TEMPLATE.format(ocr_text=ocr_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=[pil_img], return_tensors="pt"
    ).to(model.device)

    try:
        output_ids = model.generate(**inputs, max_new_tokens=2048)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM during structured extraction (%s).", tier)
        torch.cuda.empty_cache()
        return None

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids
    torch.cuda.empty_cache()  # release activation memory before next page/call

    cleaned = _strip_json_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Model output wasn't valid JSON: %s\nRaw output: %s", e, raw_text)
        return None

    try:
        return DocumentExtraction.model_validate(data)
    except ValidationError as e:
        logger.warning("Extracted JSON didn't match schema: %s\nData: %s", e, data)
        return None
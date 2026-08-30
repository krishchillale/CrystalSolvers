# OCR / Document AI Service — SIH26047

Owns: preprocessing → OCR → raw structured text output.
Hands off to: entity-extraction / structuring stage (your partner), which
consumes `OCRResult` (see `app/schema.py`) and turns it into the full
`ExtractedEntity` records for the clinical record.

## Why this design

- **Preprocessing is separated from OCR.** Bad scans are the #1 cause of bad
  OCR — cleaning the image first is cheap and model-agnostic.
- **The OCR engine is swappable.** `app/ocr_engine.py` exposes one function,
  `run_ocr(image_path) -> OCRResult`. Under the hood it tries Qwen3-VL-8B
  first (best real-world accuracy on Hindi/Devanagari text per our research)
  and falls back to Tesseract if the GPU model isn't available — so the rest
  of the pipeline (and your partner's code) never needs to know which engine
  actually ran.
- **Output is structured from the start**, not a flat string — so the
  handoff to entity extraction has page/line-level text with a place for
  confidence, matching the blueprint's `ExtractedEntity` fields
  (source_document, source_page, confidence).

## Setup

```bash
pip install -r requirements.txt

# System dependency for the Tesseract fallback path:
# sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar
```

Qwen3-VL-8B needs a GPU with ~24GB VRAM (e.g. a single RTX 3090/4090/A10G).
If you don't have that set up yet, the service will automatically fall back
to Tesseract so you can keep developing the pipeline end-to-end — just know
the fallback's accuracy is much worse on handwriting and Hindi/Marathi text.
Swap in the real model as soon as GPU access is sorted (see
`app/ocr_engine.py`, `USE_QWEN` flag).

## Run

```bash
# Preprocess + OCR a single file, print the structured result
python -m app.main sample_docs/sample_prescription.jpg

# Or run the FastAPI service
uvicorn app.server:app --reload --port 8001
```

`POST /ocr/extract` — multipart file upload, returns `OCRResult` JSON.

## What your partner needs to know

`OCRResult` (in `app/schema.py`) is the contract. It gives you:
- `full_text` — the whole page, for quick sanity checks
- `blocks` — list of text regions, each with text + approximate bounding
  box + a confidence score (0–1) + which engine produced it
- `document_id`, `page_number` — carried through so you can attach
  `source_document` / `source_page` on every `ExtractedEntity` you derive

Nothing in here does medical NER, date parsing, or schema normalization —
that's intentionally your half. This module's only job is: messy image in,
clean structured text out.

## TODO (next up)

- [ ] Swap in real Qwen3-VL-8B weights once GPU box is ready
- [ ] Test against real sample prescriptions/lab reports (not just synthetic)
- [ ] Tune deskew/denoise thresholds against our actual scan quality
- [ ] Wire `/ocr/extract` into the main FastAPI backend (or keep as
      microservice — TBD with Member 4)

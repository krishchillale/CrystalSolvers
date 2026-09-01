# OCR / Document AI Module — Technical Notes

**Project:** SIH26047 — Patient Case-Taking Software (Ministry of Ayush)
**Owns:** Module B — turning a photo/scan/PDF of a medical document into
clean, structured data.
**Hardware target tested on:** RTX 3050, 6GB VRAM, Windows.

This document explains what's been built, why each choice was made, and
what's still rough around the edges — so a teammate (or future you) can
pick this up without re-deriving all the decisions.

---

## 1. Pipeline overview

```
Image or PDF
     │
     ▼
Preprocessing  (deskew, denoise, grayscale)
     │
     ▼
OCR  (Qwen VLM primary, Tesseract fallback)
     │
     ├──► raw OCRResult  (full transcript, always complete)
     │
     ▼
Structured extraction  (second VLM pass, JSON schema)
     │
     ▼
DocumentExtraction  (clinical fields only, admin info excluded)
     │
     ▼
 <stem>.json   +   <stem>.csv
```

Two outputs, two purposes:
- **Raw OCR transcript** (`full_text` inside the `.json`) — everything on
  the page, unfiltered. Useful as an audit trail / fallback if structured
  extraction misses something.
- **Structured extraction** (`extraction` inside the `.json`, and the whole
  of the `.csv`) — only clinically relevant fields: patient info, test
  results, medications. Administrative info (hospital name, doctor name,
  phone/email/address) is deliberately excluded here, even though it's
  still visible in the raw transcript.

---

## 2. File map

| File | Role |
|---|---|
| `preprocess.py` | Deskew, denoise, grayscale — OpenCV |
| `pdf_utils.py` | Renders PDF pages to images (PyMuPDF) |
| `vlm_utils.py` | Caps image resolution before it reaches the model |
| `ocr_engine.py` | `run_ocr()` — the single entry point; picks Qwen tier or Tesseract |
| `schema.py` | `OCRResult` / `TextBlock` — the raw-OCR contract |
| `entity_schema.py` | `DocumentExtraction` — the clean, structured contract |
| `extract_structured.py` | Second VLM call: image → structured JSON |
| `export_csv.py` | `DocumentExtraction` → flat CSV rows |
| `main.py` | CLI entrypoint, wires everything together |
| `server.py` | FastAPI wrapper (`/ocr/extract`) for backend integration |

---

## 3. Preprocessing (`preprocess.py`)

Runs before OCR ever sees the image:
- Grayscale conversion
- Denoise (`cv2.fastNlMeansDenoising`)
- Deskew — detects rotation via `minAreaRect` on thresholded text pixels,
  corrects with `warpAffine`. Tested against a deliberately rotated sample
  and confirmed it recovers the angle correctly.

Two entry points: `preprocess_pipeline()` for a file path (loads from
disk), `preprocess_image()` for an already-loaded array (used for PDF
pages, which arrive as arrays, not files).

## 4. PDF handling (`pdf_utils.py`)

Uses **PyMuPDF** (`import pymupdf as fitz`), not `pdf2image`, specifically
because `pdf2image` needs the external Poppler binary installed separately
— an extra Windows setup step PyMuPDF avoids (pure pip install).

Renders each page to an image at **150 DPI** (default). This was
originally 300 DPI, but that produced images ~4x larger in pixel count,
which caused CUDA out-of-memory errors on the VLM even at the smaller
model tier — see §7.

## 5. OCR engine (`ocr_engine.py`)

Single entry point `run_ocr(img) -> OCRResult`. Internally tries, in
order, until one succeeds:

1. **Qwen3-VL-8B** (4-bit quantized) — best real-world accuracy,
   especially on Hindi/Devanagari text, per benchmark research. **Does
   NOT fit on a 6GB card** — confirmed via testing (`accelerate` can't
   auto-offload under bitsandbytes 4-bit without extra config). Left in
   the code path for when running on a bigger GPU (e.g. free-tier cloud
   GPUs have 16GB+).
2. **Qwen2.5-VL-3B** (4-bit quantized) — the tier that actually works on
   this hardware. `USE_QWEN_TIER` is set to `"3b"` by default to skip the
   guaranteed-to-fail 8B attempt and save ~30s per run.
3. **Tesseract** (CPU) — final fallback if both VLM tiers fail (no GPU,
   OOM, model load error, etc.). Configured for `hin+mar+eng`. Much
   weaker on handwriting and Indic scripts — Tesseract is designed for
   printed text only.

Every failure mode falls through gracefully — the pipeline never crashes
outright, it just degrades to a weaker engine and records a warning in
`OCRResult.warnings`.

### Why VLM over classical OCR at all

Research done before building (see conversation history) found that for
this exact use case — mixed Hindi/Marathi/English medical documents,
some handwritten — a real-scan benchmark showed most classical/OCR-VLM
systems collapse badly on real (non-clean) scans, with results spread
across a huge accuracy range. Qwen-family models held up comparatively
well and are free/self-hostable, which is why they were chosen over
PaddleOCR (couldn't even be benchmarked reliably) and DeepSeek-OCR
(catastrophic repetition failures on real scans — dangerous for medical
data, since a hallucinated repeated value could corrupt a lab result).

## 6. How the model actually reads text (mechanism, not magic)

Worth understanding because it changes how much to trust the output:

- **Tesseract**: classical pattern matching against known character
  shapes. Outputs a real per-word confidence score.
- **Qwen (VLM)**: the image is sliced into patches, run through a vision
  transformer, and converted into tokens — same conceptual space as text
  tokens. The model then **generates** its answer token-by-token,
  predicting each next word from everything it's seen (image + prompt +
  its own output so far). It's not reading characters — it's predicting
  plausible continuations, having learned from massive training data.

**Practical consequence**: it can produce a fluent, confident-looking
value that isn't actually what's on the page — especially numbers, which
have no semantic "meaning" to anchor a guess against. There's no
equivalent to Tesseract's low-confidence flag. This is why
`confidence: 0.9` in `OCRResult` is currently a **hardcoded placeholder**,
not a real measurement — flagged clearly in the code, but worth
remembering when reviewing output quality.

## 7. Structured extraction (`extract_structured.py`, `entity_schema.py`)

Rather than regex-parsing the raw OCR text, this does a **second VLM
call** on the same image, asking directly for a JSON object matching
`DocumentExtraction`'s shape. This tends to be more accurate than
post-hoc text parsing, since the model can use visual layout (which
column is which, what's grouped together) that's lost once everything is
flattened into plain text.

**Schema** (`entity_schema.py`):
```python
DocumentExtraction:
    document_type: str | None      # "lab_report", "prescription", etc.
    patient_name: str | None
    age_sex: str | None
    report_date: str | None
    investigations: list[Investigation]   # dynamic — no fixed field list
    medications: list[Medication]         # dynamic — no fixed field list
```

**Deliberately excluded**: hospital/clinic name, doctor name, phone,
email, address. These are administrative/contact fields, not clinical
data — the prompt explicitly instructs the model not to extract them,
and they're not in the schema at all (so even if the model ignored the
instruction, `pydantic` would silently drop any extra keys it returned).

**Why dynamic fields, not a rigid predefined template**: real lab reports
vary hugely even within one category (a basic CBC has ~10 parameters, a
comprehensive one has 20+). A rigid predefined field list risks silently
dropping real clinical values that weren't anticipated — a real
data-loss risk for a medical tool. Instead, `investigations`/`medications`
capture *whatever the model actually finds*, tagged by category, rather
than being constrained to a fixed list. This also matches how real EHR
systems (e.g. FHIR) model lab observations — a normalized "one row per
result" shape, not one column per possible test.

Reuses the already-loaded Qwen model (`get_qwen_for_tier()` in
`ocr_engine.py`) rather than loading it a second time — saves VRAM and
time, important given how tight 6GB is.

## 8. CSV/JSON export (`export_csv.py`, `main.py`)

Output is named after the input file — `test1.jpeg` produces `test1.json`
and `test1.csv` — and **overwrites** on each run (not an accumulating
log; one file per document).

- **`<stem>.json`**: combined output per page — both the raw `OCRResult`
  (`full_text`, preprocessing steps, engine used) and the structured
  `DocumentExtraction` (or `null` if structured extraction failed for
  that page).
- **`<stem>.csv`**: flat, "tidy long format" — one row per investigation
  or medication, with patient-level fields (name, age/sex, date) repeated
  on every row so each row is self-contained and the file opens cleanly
  in Excel with no separate lookup needed.

If structured extraction fails for a page (bad JSON from the model, CUDA
OOM, etc.), the `.csv` is **not written/overwritten** for that run — the
raw OCR JSON is still saved. This means a stale `.csv` from a previous
successful run can persist if a later run fails; check the terminal
output (`Saved <stem>.csv` vs `No successful structured extractions`) to
know which happened.

## 9. Known issues / things to watch

- **`confidence: 0.9` on Qwen results is a placeholder**, not a real
  score (see §6). Tesseract's confidence *is* real (averaged per-line
  from its own word-level confidences).
- **Large images cause CUDA OOM** on a 6GB card — mitigated by capping
  images to 1280px on the longest side (`vlm_utils.py`) before they reach
  the model, and rendering PDFs at 150 DPI rather than 300. If OOM still
  happens on an unusually large/dense document, this cap is the first
  place to check.
- **`torch.cuda.empty_cache()`** is called after every generate call
  (success or failure) in both `ocr_engine.py` and `extract_structured.py`
  to avoid memory fragmenting across multiple pages of the same PDF.
- **Windows-specific setup gotchas hit so far**: CPU-only PyTorch installs
  silently if you don't use the CUDA-specific install command from
  pytorch.org; `bitsandbytes` needs a matching CUDA-compiled torch;
  Tesseract needs `tesseract_cmd` pointed at its `.exe` explicitly since
  Windows doesn't reliably put it on PATH; `AutoModelForVision2Seq` is
  deprecated in newer `transformers` — use `AutoModelForImageTextToText`.
- **Handwriting accuracy hasn't been separately tested yet** — all real
  testing so far has been on printed/scanned documents (a lab report, a
  synthetic prescription). Per earlier research, expect meaningfully
  worse accuracy on handwriting even from the VLM tiers.
- **Vocabulary normalization not yet handled**: the same test can be
  named differently across labs ("Hb" vs "Hemoglobin" vs "HGB"). Not yet
  addressed — worth deciding whether to normalize in the prompt or in a
  post-processing step before this goes further.

## 10. What's next / not yet built

- Report-category detection (blood/urine/lipid panel/etc.) — discussed,
  not yet implemented.
- Bounding boxes / source-region evidence for structured fields — needed
  for the blueprint's "physician-verifiable, evidence-backed" requirement,
  not yet built (Qwen's plain transcription mode doesn't return them;
  would need grounding-mode prompting or a separate layout-detection
  pass).
- Testing against real Hindi/Marathi documents specifically (testing so
  far has been English-language documents).
- Handoff to the entity-extraction/structuring teammate — `OCRResult`
  (raw) is the contract for that; `DocumentExtraction` (this doc's §7) is
  a fast local-dev shortcut, not necessarily the final backend contract.

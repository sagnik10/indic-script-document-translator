# Pretrained Hindi and Gurmukhi Handwritten Page OCR

This is a self-contained, inference-only application for PDF, DOCX, JPG, JPEG,
and PNG documents. Native PDF and Word text is extracted directly without OCR.
Scanned pages use fast local Punjabi/Hindi/English OCR, with selective neural
handwriting refinement available only for unresolved lines. Validated Hindi or
Punjabi text is translated to English and exported as a Word document.

## Streamlit upload interface

From the repository root on Windows, install the dependencies and launch the
browser interface:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\pretrained_page_ocr\requirements-ui-windows.txt
.\.venv\Scripts\python.exe -m streamlit run .\pretrained_page_ocr\streamlit_app.py
```

Open the displayed local URL, upload a PDF/DOCX/JPG/JPEG/PNG document, choose
**Fast document** or **Hybrid handwriting accuracy**, and select **Transcribe
and translate document**.
The primary download is `translated_en.docx`. It contains one paragraph for
every detected line in reading order; a line that cannot pass the safety gates
is retained as an explicit review entry instead of being silently dropped or
invented. Additional TXT, JSON, CSV, and image artifacts are available in a
collapsed diagnostics section. Uploaded data is processed in a temporary
directory and deleted after the result bytes have been collected.

**Fast document** directly extracts native text and uses Tesseract only on
scanned pages. **Hybrid handwriting accuracy** keeps that route, then sends
only unresolved handwritten scan lines to the neural recognizers. It does not
run a large vision model on every PDF page. CUDA remains the strongest speed
improvement for handwriting refinement.

A scanned PDF is still a collection of page images, so it is not inherently
cheaper than a JPG. The speedup comes from native-text detection, one-pass
Tesseract OCR, modest 150-DPI rendering, and selective neural fallback.

It does **not** train or fine-tune any model, create checkpoints, save model
weights, or package model archives. Model files are downloaded from their
public repositories into the normal Hugging Face/Kaggle cache.

## Pretrained models used

| Purpose | Model | Role and limitation |
|---|---|---|
| Hindi handwriting | [`paudelanil/trocr-devanagari-2`](https://huggingface.co/paudelanil/trocr-devanagari-2) | Required Devanagari TrOCR recognizer, loaded with `TrOCRProcessor` and `VisionEncoderDecoderModel`. Its repository is associated with the IIIT Indic Hindi handwriting dataset and is word-oriented, so clearly separated word crops are recombined into lines. |
| Gurmukhi handwriting fallback | [`datalab-to/surya-ocr-2`](https://huggingface.co/datalab-to/surya-ocr-2) | Multilingual OCR fallback with handwritten-note examples and documented Punjabi (`pa`) evaluation. This is **not represented as a dedicated Gurmukhi handwriting checkpoint**. Every result must pass Gurmukhi Unicode purity and quality checks. |
| Translation | [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | Contextual translation using `hin_Deva` or `pan_Guru` to `eng_Latn`. Adjacent validated scan lines may be grouped into short source units. Only accepted source OCR is translated. |
| Last-resort local OCR | Official Tesseract `hin` and `pan` language data | Used only when the corresponding neural recognizer cannot load. Tesseract is not claimed to be handwriting-specialized. |

The public Surya multilingual benchmark lists Punjabi support. A public model
whose repository only classifies isolated Gurmukhi characters is deliberately
not used as a full-line HTR model.

### Model-license note

Review the model licenses before commercial deployment. The Hindi repository
declares MIT. Surya's code is Apache-2.0, while its model card describes a
modified OpenRAIL license with separate terms for larger commercial users.
NLLB-200 is CC-BY-NC-4.0 and its model card describes it as a research model,
not a certified or domain-specialist legal/medical translation system.

## Kaggle quick start

1. Create a Kaggle notebook and enable **GPU** under *Notebook options*.
2. Add the page image as a Kaggle input dataset.
3. Upload or paste [`kaggle_app.py`](kaggle_app.py) into the notebook working
   environment.
4. Run this installation cell once:

```bash
!sudo apt-get update -qq && sudo apt-get install -y -qq \
    tesseract-ocr tesseract-ocr-pan tesseract-ocr-hin
%pip install -q -r /kaggle/working/pretrained_page_ocr/requirements-kaggle.txt
```

If you copied only the Python file rather than the directory, install directly:

```bash
%pip install -q "transformers>=5.12.1,<5.17" "surya-ocr==0.22.1" \
    "opencv-python-headless==4.11.0.86" "pillow>=10.2,<11" \
    "sentencepiece>=0.2,<1" "protobuf>=5,<7" "pytesseract>=0.3.13,<1" \
    "python-docx>=1.1,<2"
```

Kaggle already includes a CUDA-enabled PyTorch build. Avoid replacing it unless
the installed version is below the requirement, because reinstalling PyTorch is
large and can break the notebook's CUDA compatibility.

Near the top of `kaggle_app.py`, set the one normal input value:

```python
PAGE_IMAGE = "/kaggle/input/my-handwritten-pages/test.jpg"
```

Then run:

```bash
!python /kaggle/working/pretrained_page_ocr/kaggle_app.py
```

An explicit command-line path takes priority over `PAGE_IMAGE`:

```bash
!python /kaggle/working/pretrained_page_ocr/kaggle_app.py \
    --image /kaggle/input/my-handwritten-pages/test.jpg \
    --output /kaggle/working/page_ocr_result \
    --mode fast_cpu
```

When `PAGE_IMAGE` is `None` (or still contains the `...` example placeholder),
the program can select the first likely JPG, JPEG, or PNG below
`/kaggle/input`. An explicit path is recommended when a dataset contains more
than one image.

## Processing flow

1. Apply EXIF orientation and conservative optional orientation correction.
2. Detect a document boundary, rectify perspective, and deskew when justified.
3. Keep the corrected RGB page plus illumination-corrected, CLAHE, mildly
   denoised grayscale, and adaptive-threshold variants.
4. Detect logical text lines with Surya's pretrained detector; fall back to
   OpenCV connected-component and projection grouping if necessary.
5. Run Hindi and Gurmukhi recognition candidates and score them using model
   confidence, Unicode-script purity, source-letter count, and text quality.
6. Mark ambiguous, empty, mixed-script, short, low-purity, or low-confidence
   lines for review instead of presenting them as reliable.
7. Normalize only unambiguous Indic combining-mark spacing, then protect dates,
   IDs, amounts, phone numbers, URLs, emails, and existing English spans.
8. Group up to three adjacent validated scan lines for translation context and
   translate with bounded quality beam search (four beams in fast mode, six in
   hybrid mode). Validate English shape, repetition, length, and protected-token
   restoration before accepting the result.
9. Map every accepted contextual translation back to its source lines and save
   per-line audit data and visual detection overlays.

For PDF/DOCX input, the fast document router runs before this handwriting path:

1. Extract selectable PDF or Word text directly.
2. Preserve existing English, numbers, identifiers, tables, headers, footers,
   images, and native Word structure.
3. Render only image-only PDF pages and OCR them once with `pan+hin+eng`.
4. In hybrid mode, refine only unresolved logical handwriting lines with the
   neural recognizers.

Inference uses `torch.inference_mode()`, lazy single-copy model loading, batched
line recognition and translation, GPU autocasting where safe, and GPU-memory
release between detector, recognizer, and translator stages.

The loader disables meta-device/low-memory initialization for these checkpoints,
resolves their declared tied weights, and verifies that no parameter remains on
the `meta` device before inference. It never uses `to_empty()` because doing so
could silently materialize random OCR weights. If a checkpoint is incomplete,
the recognizer is rejected and the documented Tesseract/review fallback is used.

## Output

The default directory is `/kaggle/working/page_ocr_result/`:

```text
page_ocr_result/
├── corrected_page.jpg
├── detected_lines.jpg
├── line_crops/
│   ├── line_001.jpg
│   └── ...
├── transcription.txt
├── translation_en.txt
├── translated_en.docx
├── result.json
└── result.csv
```

`result.json` and `result.csv` retain each line's bounding box, selected script,
source text, Unicode-script purity, OCR confidence where available, recognition
provider/model, review reasons, translation status, translation confidence,
translation QA score, context-group identifier, and English output.
Accepted and review-required lines use different colors in
`detected_lines.jpg`.

## Reliability and limitations

- Handwriting OCR is not guaranteed correct. Historical, faded, overwritten,
  highly cursive, stamped, or unusual handwriting may require human review.
- Surya OCR 2 is a verified multilingual fallback with Punjabi coverage, not a
  dedicated Gurmukhi handwriting model. The console and audit output report
  this distinction explicitly.
- The Hindi checkpoint is word-oriented. Line splitting improves compatibility
  but irregular touching handwriting can still reduce recognition quality.
- Translation cannot repair unreliable OCR. A line that fails source-script and
  quality validation remains untranslated and is marked `review_required`.
- NLLB's own model card says it is not intended as a production document,
  legal, or medical translation system. Use qualified human review for those
  contexts and comply with its non-commercial license.
- Empty or corrupt model output is never replaced with a generic English
  sentence. A per-line translation failure does not discard the OCR result.
- Printed forms, ruled lines, signatures, seals, marginal notes, and overlapping
  handwriting remain difficult cases for automatic line detection.
- First execution needs internet access and several gigabytes of cache space for
  the OCR and translation models. Later executions reuse the Hugging Face cache.
- Complete cached Hugging Face models are loaded locally first. If a model is
  not cached, an optional `HF_TOKEN` increases Hub download rate limits; it is
  not a paid API requirement.

## Local model-free tests

The tests exercise Unicode analysis, source-quality gates, candidate selection,
spatial merging/reading order, and OpenCV fallback detection without loading or
downloading neural models:

```powershell
python -m pytest tests/test_pretrained_page_ocr.py -q
```

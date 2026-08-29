# Difficult-Document Punjabi/Hindi Translator

A modular Python 3.11+ Streamlit application for local extraction, review, English translation, and format-preserving reconstruction of native PDFs, degraded scans, photocopies, mobile photographs, and DOCX files. Punjabi Gurmukhi and Hindi Devanagari are the priority source languages; existing English and structured identifiers are preserved.

This is a realistic AI-assisted MVP, not a certified OCR, medical, legal, handwriting, or translation system. Raw OCR, alternatives, corrections, inferred readings, unreadable regions, translations, coordinates, and layout decisions remain auditable.

## Primary workflow

```text
upload and validate
  -> page detection / native extraction
  -> boundary crop and perspective correction for photographs
  -> quality analysis and adaptive preprocessing
  -> border/noise masking and coarse text-line detection
  -> pixel-based printed / handwriting / table / stamp / signature classification
  -> visual script candidate + expected-language prior resolution
  -> script-specific printed OCR or language-specific HTR
  -> meaningful Unicode script evidence and final script resolution
  -> line reconstruction, confidence filtering, and strict source-language validation
  -> missing-span detection and tightly bounded source-language-only reconstruction
  -> optional human review of uncertain OCR and reconstruction candidates
  -> protected terminology and validated Punjabi/Hindi-to-English translation
  -> collision-aware in-place replacement on the canonical source template
  -> geometry/preservation validation, primary-document preview, and separate JSON audit
```

The app processes PDF pages one at a time and releases page raster intermediates after they have been reduced to typed metadata and bounded previews. Heavy OCR, HTR, language, reconstruction, and translation models are lazy-loaded once by the central model manager. CUDA is selected when available and CPU is the automatic fallback.

## Features

- Upload PDF, scanned PDF, DOCX, PNG, JPG, or JPEG from a wide Streamlit UI.
- Validate extension, signature/container, size, readability, and sanitized filename without executing uploads or overwriting originals.
- Route native PDF/DOCX text directly while applying OCR only to scanned, sparse, image-heavy, embedded-image, or force-OCR content.
- Preserve page dimensions, page order, source images, tables, rules, boxes, headers, logos, photographs, stamps, seals, signatures, watermarks, form branding, margins, and graphical marks where the source format exposes them.
- Keep every OCR line and word associated with page coordinates, a region, engine, confidence, alternatives, parent relationship, and reading order.
- Detect Gurmukhi, Devanagari, Latin, digits, and mixed-script spans. Punjabi maps to `pa`/`pan_Guru`; Hindi/Devanagari routes to the supported Devanagari language classifier result; English is not translated.
- Generate rectified grayscale, CLAHE, illumination-corrected, denoised, sharpened, and mild-threshold candidates without replacing thin handwriting with one destructive binary image. Run the ensemble only on logical lines/paragraphs.
- Route scripts from enhanced grayscale pixels before OCR. Visual page/region candidates, OCR-derived Unicode scripts, and resolved scripts are stored separately; failed ASCII OCR cannot overwrite visual evidence.
- Mask dense photocopy edge bands, binding artifacts, sustained page-border rules, and speckle before handwriting/script statistics. Thresholded images remain optional OCR candidates rather than authoritative visual inputs.
- Score handwriting from connected-component variation, line curvature, baseline/stroke/spacing irregularity, connectedness, and page context without depending on successful OCR. Handwriting-heavy pages use line-level grouping and HTR routing.
- Allow an `Auto`, `Punjabi`, `Hindi`, or `Punjabi + Hindi` expected-language routing prior. It selects recognition routes but never bypasses Unicode-script/source-quality validation.
- Give isolated `|`, `I`, `£`, `?`, `eee`, `Qw`, repeated-symbol, and other non-linguistic OCR fragments essentially zero dominant-script voting weight.
- Resolve `pa`/`pan`/`pan_Guru` and `hi`/`hin`/`hin_Deva` aliases centrally. `und`, `unknown`, `mul`, punctuation fragments, script-inconsistent ASCII, and low-quality OCR are never sent to translation.
- Abort automatic translation for a page when the configured catastrophic unreadable ratio is exceeded. The enhanced page, original raster regions, review table, and audit remain available.
- Keep printed OCR and handwriting recognition behind separate interfaces. TrOCR models load per language only when explicitly configured and handwriting OCR is enabled.
- Never run an English handwriting checkpoint as a silent Punjabi/Hindi fallback. Missing HTR models produce `UNREADABLE`, preserve the handwriting image, and show setup guidance.
- Classify regions conservatively as `printed_text`, `handwriting`, `table_form`, `stamp_seal`, `signature`, `graphical_content`, or `unknown`. Heuristic classifications and confidence are auditable and not treated as ground truth.
- Preserve signatures, stamps, seals, and graphics as source imagery and never send them to normal OCR, translation, or replacement rendering.
- Detect explicit missing markers, replacement glyphs, repeated punctuation, malformed boundaries, partial words, script inconsistencies, abrupt gaps, low-confidence OCR, and OCR-candidate disagreement at logical line level.
- Reconstruct only a bounded Punjabi Gurmukhi or Hindi Devanagari source span after readable-character, context-quality, validated-neighbor, protected-entity, script-purity, and final source-quality gates pass. The reconstruction provider cannot return an English translation or rewrite the complete sentence.
- Auto-accept source candidates only at confidence `>= 0.90`; route `0.70-0.90` candidates to crop-and-context review; preserve `[unclear]` below the review threshold. Names, dates, amounts, numbers, references, medicines, dosages, legal sections, phone numbers, addresses, and signatures are excluded from contextual inference.
- Distinguish `OCR_EXTRACTED`, `OCR_CORRECTED`, `MODEL_INFERRED`, `MANUALLY_CONFIRMED`, and `UNREADABLE` provenance. Automatically inferred source text always remains visibly marked in review mode and the audit.
- Prohibit contextual model guesses for high-risk medical/legal/government contexts and for nearby diagnoses, drugs, dosages, dates, numbers, provisions, names, references, or monetary values.
- Pause before translation for editable low-confidence review. Review is mandatory when the configured HTR providers cannot read a detected Punjabi/Hindi handwriting line. Only rows explicitly checked by the user change downstream source text; raw OCR remains unchanged in the audit.
- Protect dates, phone numbers, currency, URLs, email addresses, dosages, case/file/dispatch references, hospital/department names, abbreviations, codes, user-protected terms, and glossary entries with placeholders before translation.
- Translate contextual chunks in batches with a cached provider. The default open model is `facebook/nllb-200-distilled-600M`, an NLLB multilingual transformer supporting Punjabi and Hindi to English. Compatible local providers can replace it.
- Fit selectable/searchable English using wrapping, measured font reduction, line-spacing adjustment, safe expansion, and collision detection.
- Make the source-template reconstruction the only primary output: untranslated, unreadable, unsafe, or overflowing regions remain visually unchanged instead of being moved to annotations or appended report pages.
- Produce `originalname_translated_en.pdf` for PDF/images or `originalname_translated_en.docx` for Word files. The optional JSON audit is available only inside the diagnostics download section.
- Show the translated document as the primary preview. Original, enhanced, routing, and debug-bounding-box views remain inside the collapsed diagnostics area with confidence statistics, uncertain-text tables, warnings, model status, and the optional audit download.

## Adaptive preprocessing

The system measures brightness, contrast, Laplacian sharpness, noise residual, illumination/shadow variation, background ink/noise, skew, page-border confidence, perspective distortion, resolution, and connected-component irregularity before selecting transformations.

Profiles in `config/default.yaml` and the UI:

| Profile | Intended use |
|---|---|
| `auto` | Select a measured profile per page |
| `clean_scan` | Minimal enhancement; avoids degrading good text |
| `photocopy` | Background cleanup, denoising, CLAHE, guarded adaptive thresholding |
| `mobile_photo` | Page-border crop, perspective correction, shadows, skew, local contrast |
| `faded_document` | Stronger CLAHE/local enhancement and cautious sharpening |
| `handwriting_heavy` | Preserves stroke variation and avoids destructive binarization |

Available operations include EXIF/orientation handling, document-boundary detection, four-point perspective correction, crop, same-canvas deskew, illumination division, shadow correction, grayscale normalization, CLAHE, adaptive denoising, guarded adaptive thresholding, morphological photocopy cleanup, edge-preserving sharpening, border cleanup, and measured 1x-4x OCR upscaling.

Perspective correction handles planar pages photographed at an angle. Folded, curled, severely warped, occluded, or finger-covered pages are not fully dewarped; local contrast/shadow correction may help, but manual recapture or specialist dewarping can still be necessary.

## Architecture and modules

```text
app.py
config/default.yaml
translator_app/
  schemas.py                     typed Document, Page, Region, TextLine, TextBlock
  pipeline.py                    analyze/review/finalize orchestration
  temporary_storage.py           task-scoped secure cache and cleanup
  report_generator.py            flat OCR/translation/layout JSON audit
  models/model_manager.py        lazy model registry and CPU/CUDA routing
  core/
    document_loader.py           validated format router
    pdf_processor.py             native/scanned/mixed page extraction
    docx_processor.py            paragraphs, runs, tables, images, headers/footers
    quality_analysis.py          image metrics and auto profile selection
    preprocessing_profiles.py    named profile definitions
    image_processor.py           perspective, shadow, cleanup, enhancement
    layout_detection.py          region classes, tables, critical graphics
    region_merging.py            baseline/overlap line-region grouping
    block_grouping.py             OCR fragment and validated paragraph grouping
    script_detection.py          shared Unicode-script ratios and priors
    visual_routing.py            pixel-first script/handwriting evidence and border masking
    source_validation.py         canonical language resolution and translation gates
    layout_analyzer.py           native/PDF classification and OCR deduplication
    printed_ocr.py               region-aware low-confidence multi-pass OCR
    handwriting_ocr.py           line segmentation, HTR routing, review fallback
    htr_providers.py             capability-declared local/Hugging Face HTR providers
    ocr_engine.py                coordinate-aware Tesseract provider
    ocr_ensemble.py              candidate comparison and alternatives
    page_ocr.py                  per-page OCR/HTR/read-order coordinator
    confidence_analysis.py       uncertainty/provenance state transitions
    ocr_normalization.py         non-semantic Unicode/OCR cleanup
    language_detector.py         Unicode script plus classifier routing
    missing_text_detector.py     bounded gap detection and context/protected-entity gates
    source_reconstruction.py     source-script-only span prediction and threshold policy
    context_engine.py            backward-compatible reconstruction facade
    terminology.py               values, terms, and glossary protection
    translation_engine.py        provider interface, chunking, batching, cache
    layout_engine.py             text wrapping/fitting/expansion
    collision_detection.py       safe-space and critical-region checks
    reconstruction_engine.py     PDF/DOCX backend router
    renderer.py                   original/enhanced/debug/output previews
    output_validation.py         output reopening and structure checks
  reconstruction/
    pdf_reconstructor.py         localized in-place searchable PDF reconstruction
    docx_reconstructor.py        run/table/style-preserving Word output
  ui/
    app.py                       Streamlit state and two-phase review
    components.py                controls, previews, review table, summary
  utils/
    validation.py                untrusted upload validation
    file_utils.py                safe names, non-overwriting writes, temp helper
    text_utils.py                normalization, chunking, structured patterns
tests/
  integration/                   private difficult-document fixture scaffolding
```

`TextBlock` exposes page/block identity, bounding box, region type, raw/source and normalized text, region visual script, recognized Unicode script, resolved script/language, script-resolution reason, linguistic-evidence score, OCR engine/confidence/alternatives, handwriting and uncertainty flags, reconstruction type/text/confidence, missing-span detection and bounding box, reconstruction candidate/method/status, readable-character ratio, validated-context token count, protected-entity gate, protected tokens, English translation/status, font size/style, rotation, source/output coordinates, layout status, relationships, metadata, and provenance. Page records separately retain visual page script, OCR page script, resolved page script, expected-language prior, handwriting probability, page type, route counts, and noise rejections.

## Supported languages

Priority routes:

- Punjabi Gurmukhi printed OCR: Tesseract `pan`; translation source `pan_Guru` through the provider map.
- Hindi Devanagari printed OCR: Tesseract `hin`; translation source `hin_Deva`.
- English printed text: Tesseract `eng`; preserved without translation.

Configured OCR extensions include Bengali, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, and Odia. The strict automatic translation gate currently admits validated Punjabi and Hindi only; additional translation routes can be added by extending the provider and source-validation policy deliberately.

Mixed English/Punjabi/Hindi is detected at block and token-span level. English spans and structured tokens are protected while the regional-language context is translated.

## Prerequisites

- Python 3.11 or newer.
- Tesseract OCR 5.x and required language packs.
- Sufficient RAM/disk for the selected translation and optional handwriting models. The default 600M translation model is substantially slower on CPU.
- Internet access once for model downloads, or pre-populated local Hugging Face caches.
- Optional NVIDIA CUDA-compatible PyTorch installation.

## Installation

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, install the command supplied by <https://pytorch.org/get-started/locally/> first, then install the remaining requirements.

### Tesseract and Indic language packs

Tesseract is a system dependency, not a Python wheel. Official installation guidance: <https://tesseract-ocr.github.io/tessdoc/Installation.html>.

Ubuntu/Debian example:

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-pan tesseract-ocr-hin \
  tesseract-ocr-ben tesseract-ocr-mar tesseract-ocr-guj tesseract-ocr-tam \
  tesseract-ocr-tel tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ori
tesseract --list-langs
```

On Windows, install Tesseract and place `eng.traineddata`, `pan.traineddata`, `hin.traineddata`, and other required files in `tessdata`. The app automatically checks `PATH`, `C:\Program Files\Tesseract-OCR\tesseract.exe`, common per-user installation locations, and the project-local `.runtime\tessdata` language directory. Explicit overrides remain available:

```powershell
$env:DTX_TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:DTX_TESSDATA_DIRECTORY = ".runtime\tessdata"
```

Verify:

```powershell
tesseract --version
tesseract --list-langs
```

The app names missing requested packs. It uses installed requested packs when possible and stops with a concise setup message when no usable OCR pack is available.

### Translation model

The default model downloads and caches on the first regional-language translation. Pre-download with the Hugging Face CLI:

```bash
hf download facebook/nllb-200-distilled-600M
```

CLI documentation: <https://huggingface.co/docs/huggingface_hub/guides/cli>.

For an offline deployment, pre-populate the cache or configure a compatible local NLLB directory and set:

```powershell
$env:DTX_HF_LOCAL_FILES_ONLY = "true"
$env:DTX_TRANSLATION_MODEL = "D:\models\nllb-indic-en"
```

The translation provider is an interface. IndicTrans2 or another local provider can be added without changing document/OCR/reconstruction modules; the shipped default uses standard Transformers NLLB because it runs without a paid API or custom remote code.

### Source reconstruction model

Missing-span detection is always available and does not require a large model. Candidate prediction is disabled by default (`RECONSTRUCTION_MODEL: null`), so a missing span remains `[unclear]` unless a script-consistent OCR alternative supplies a sufficiently confident bounded candidate or the user confirms the source reading.

For automatic prediction, configure a local or cached Hugging Face masked-language model that has been evaluated for the required Punjabi Gurmukhi and/or Hindi Devanagari source text:

```powershell
$env:DTX_RECONSTRUCTION_MODEL = "D:\models\validated-indic-masked-lm"
$env:DTX_HF_LOCAL_FILES_ONLY = "true"
```

The adapter asks the model for one short masked span only. It rejects English/wrong-script output, digits, candidates over the configured token limit, whole-line rewrites, protected contexts, and reconstructed lines that fail source-quality validation. It never sends an image or missing span directly to English generation.

### Handwriting models

Handwriting routing is enabled by default in the UI. Providers explicitly declare supported source languages, scripts, and whether they return sequence confidence. `config/default.yaml` intentionally configures only the general English TrOCR checkpoint and leaves Punjabi/Hindi models unset:

```yaml
HTR_PROVIDERS:
  - provider_id: gurmukhi_htr
    backend: transformers_vision_encoder_decoder
    model_id: null  # local directory or verified Hugging Face model ID
    supported_languages: [pa]
    supported_scripts: [gurmukhi]
    confidence_capability: sequence_probability
    handwriting_validated: false  # change only after deployment-specific HTR evaluation
```

Set `model_id` only to a line-level `VisionEncoderDecoderModel` checkpoint independently validated for Gurmukhi handwriting, then explicitly set `handwriting_validated: true`. Until both are present, the provider is discoverable in model status but cannot receive a line image. A generic English TrOCR model is rejected if declared as Punjabi/Hindi. The public `Khalsa-Phulwari/gurumkhi-recognizer` checkpoint is deliberately not used because its model card describes isolated 41-character classification, not line transcription. `HarsimarSingh/swinv2-ocr-finetuned-panjabi` is also not enabled: its model card omits the dataset/intended-use details and reports CER 1.0, which is not acceptable for this safety-sensitive path.

For automatic Gurmukhi handwriting recognition, use the self-contained free-GPU workflow in [`training/gurmukhi_htr/README.md`](training/gurmukhi_htr/README.md). It fine-tunes a source-language checkpoint on the public IIIT Gurmukhi corpus plus composed line examples, saves resumable checkpoints, and emits a signed-by-checksum bundle only after held-out gates are evaluated. Install the downloaded bundle with:

```powershell
.\.venv\Scripts\python.exe training\gurmukhi_htr\install_model.py "C:\path\to\gurmukhi_htr_model.zip"
```

Validated local bundles under `.runtime/models` are discovered automatically on the next app restart; the YAML does not need to be edited. A bundle whose evaluation fails remains visible in model status but cannot receive handwriting regions.

Every HTR output remains Punjabi/Hindi source text until Unicode script consistency and source-quality validation pass. Direct image-to-English provider configurations are rejected. If a provider is missing, incompatible, reports low confidence, or emits the wrong script, each line crop is retained in memory for the review UI, the page imagery remains unchanged, and the line is marked `HTR_UNAVAILABLE` or `HTR_LOW_CONFIDENCE`. `HTR_UNAVAILABLE` forces the app to pause at source review even if optional low-confidence review was disabled. Compare each crop with the source, enter the exact Gurmukhi/Devanagari transcription, and explicitly confirm any lines you can read. Confirmed lines are revalidated and translated; unconfirmed lines remain visually unchanged. At least one credible line must translate and render successfully before the app exposes a primary translated-document download. A zero-translation source-preserved copy is never labeled or downloaded as a translated document. Manual readings are recorded with `MANUALLY_CONFIRMED` source provenance plus a `MANUALLY_CORRECTED` processing event; raw HTR/OCR evidence is never overwritten.

`HANDWRITING_MODELS` remains accepted as a backward-compatible shorthand when `HTR_PROVIDERS` is absent. Large models are lazy-loaded only for matching language/script regions. `HF_LOCAL_FILES_ONLY: true` requires the selected checkpoint to exist in the Hugging Face cache or at a local path.

## Configuration

Main settings in `config/default.yaml`:

- `MAX_UPLOAD_SIZE`
- `OCR_DPI`, `OCR_UPSCALE_FACTOR`, `OCR_LANGUAGES`, `PRINTED_OCR_PSM_CANDIDATES`
- `OCR_LOW_CONFIDENCE_THRESHOLD`, `HANDWRITING_CONFIDENCE_THRESHOLD`
- `RECONSTRUCTION_ACCEPT_THRESHOLD`, `AUTO_RECONSTRUCT_THRESHOLD`, `REVIEW_RECONSTRUCT_THRESHOLD`
- `MIN_CONTEXT_QUALITY`, `MIN_VALIDATED_CONTEXT_TOKENS`, `MAX_RECONSTRUCTION_SPAN_TOKENS`
- `RECONSTRUCTION_MODEL`, `DOCUMENT_DOMAIN`
- `MIN_OUTPUT_FONT_SIZE`, `ENABLE_SAFE_BLOCK_EXPANSION`, `MAX_BLOCK_EXPANSION_POINTS`
- `TRANSLATION_BATCH_SIZE`, `TRANSLATION_MODEL`, maximum input characters
- `PREPROCESSING_PROFILE`, `MIN_REGION_AREA`, `ENABLE_TABLE_DETECTION`
- `MIN_REGION_WIDTH`, `MIN_REGION_HEIGHT`, `MIN_OCR_CHARACTER_COUNT`, `MIN_SOURCE_LETTER_COUNT`
- `MIN_TEXT_QUALITY_SCORE`, `MIN_SOURCE_SCRIPT_RATIO`, `CATASTROPHIC_UNREADABLE_RATIO`
- `RECONSTRUCTION_MIN_READABLE_RATIO`, `HANDWRITING_PAGE_THRESHOLD`, region/paragraph merge ratios
- `EXPECTED_SOURCE_LANGUAGE`, `VISUAL_SCRIPT_MIN_CONFIDENCE`, `VISUAL_INDIC_HEADLINE_THRESHOLD`
- `HANDWRITING_HEAVY_THRESHOLD`, `BORDER_NOISE_MAX_FRACTION`
- `RECONSTRUCTION_MODE`, `BACKGROUND_RELIABILITY_THRESHOLD`
- `PRESERVE_UNREADABLE_HANDWRITING_AS_IMAGE`, `HANDWRITING_LANGUAGE_HINT`, `HTR_PROVIDERS`, `LOCAL_HTR_MODEL_DIRECTORY`
- `MAX_PREVIEW_PAGES`, `TEMP_DIRECTORY`, `OUTPUT_DIRECTORY`
- `DEVICE`, `HF_LOCAL_FILES_ONLY`, `LOG_LEVEL`, `DEBUG`, `LOG_DOCUMENT_TEXT`

Override a setting with `DTX_<NAME>` or point `DTX_CONFIG` to another YAML file:

```powershell
$env:DTX_DEVICE = "cpu"
$env:DTX_PREPROCESSING_PROFILE = "photocopy"
$env:DTX_CONFIG = "C:\secure-config\translator.yaml"
```

## Run

From the repository root in the activated environment:

```powershell
streamlit run app.py
```

1. Upload the source.
2. Choose the expected source-language prior, OCR packs, automatic/manual preprocessing, upscaling, printed/handwriting toggles, risk profile, thresholds, glossary, and protected terms.
3. Select **Analyze document**.
4. Inspect original, enhanced, and review-overlay pages. Enable **Routing debug** to compare visual script, OCR-derived script, resolved script, handwriting probability, selected engines, rejected noise, and per-region validation.
5. For each uncertain handwritten line or missing-span candidate, compare the displayed crop and neighboring source-language lines. Enter only the Punjabi/Hindi source transcription and explicitly confirm credible readings. Automatically accepted `MODEL_INFERRED` spans remain separately highlighted for visual checking.
6. Select **Confirm selected source transcriptions, translate, and render**.
7. Compare original and English output and download the format-preserved translated document. Open **Diagnostics** only when you need the optional JSON audit or routing details.

Disable the review pause for a one-action analyze/finalize flow.

## Reconstruction behavior

The primary reconstruction always uses the uploaded page/document as its canonical visual template. Only source-validated, successfully translated Punjabi/Hindi regions are eligible for replacement. Text fitting starts at the source font size, wraps within the source box, reduces size and line spacing conservatively, and attempts only collision-safe local expansion. If fitting, background neutralization, or collision checks fail, the renderer leaves the original region unchanged.

For scanned PDF/image input, the original raster/vector page remains the base layer and successful English replacements remain selectable. For DOCX input, replacements occur at the smallest safe paragraph, run, table-cell, header, or footer reference while retaining the original package, images, styles, sections, margins, and alignment. No transcript, annotation table, audit page, debug overlay, or continuation page is appended to the primary document.

## Audit report

The JSON contains document/page/region metadata and flat block records with:

- page, block, and region IDs and coordinates;
- expected language, visual page/region script, OCR-derived Unicode script, resolved script/language, resolution reason, handwriting probability/page type, engine routes, and linguistic-evidence score;
- region type and preservation state;
- raw OCR, normalized text, script, language, and language confidence;
- OCR engine, confidence, word metadata, alternatives, and provenance;
- handwriting and uncertainty flags;
- reconstruction type, detected-span coordinates, candidate, method, confidence/status, readable-character ratio, validated-context token count, protected-entity gate, and whether a human confirmed it;
- protected tokens, English translation, and translation status;
- source/output coordinates, render mode, and layout status.

The audit intentionally contains sensitive document content and must be protected like the source file.

## Tests and verification

Unit and local integration tests do not download translation or handwriting models:

```powershell
python -m pytest
python -m pytest --cov=translator_app --cov-report=term-missing
python -m compileall -q app.py translator_app tests
python -c "import app; from translator_app.pipeline import DocumentTranslationPipeline; print('imports ok')"
```

Streamlit smoke test:

```powershell
python -m streamlit run app.py --server.headless true --server.port 8501
```

Private difficult-document scaffolding is in `tests/integration/`. Place de-identified fixtures in `tests/fixtures/difficult_documents/` using the documented names, then run:

```powershell
python -m pytest -m integration
```

Absent private fixtures are skipped. Tests never insert fake OCR text to simulate success.

## Privacy and security

- Local libraries/models are the default; no uploaded content is sent to a paid/cloud API.
- Extensions, signatures, MIME/container structure, size, and paths are validated.
- Uploaded files are never executed or overwritten.
- Temporary cache directories are randomized, application-controlled, permission-restricted where supported, and recursively removed on context exit.
- Page raster work is independent and preview count is bounded.
- Outputs are downloaded from memory. Optional persistence uses exclusive creation and safe names.
- Module logs contain identifiers, counts, stages, quality metrics, model names, and exception types, not full source text by default.
- Remote/custom providers require an explicit deployment choice and privacy review.

## Graceful degradation

- CUDA missing: CPU fallback.
- Tesseract executable or language pack missing: concise installation guidance rather than an unhandled crash.
- Printed OCR disabled: native extraction continues and scanned content remains visually preserved.
- Punjabi/Hindi HTR checkpoint missing: handwriting is `UNREADABLE`, visually preserved, and reviewable.
- Reconstruction model missing: detection and safe normalization continue; missing/damaged text stays `[unclear]`, flagged, untranslated, and reviewable.
- Translation model unavailable/offline/out of memory: affected blocks are marked failed and source content remains.
- Unreliable background, critical overlap, or no collision-free fit: the original source region remains unchanged and the issue is recorded only in diagnostics.
- Unresolvable DOCX structure: original content remains unchanged and warnings identify the affected region.

## Important limitations

- OCR accuracy depends on scan resolution, script pack, typography, blur, noise, shadow, skew, segmentation, and damage. It is not guaranteed.
- Handwriting recognition is model/language/domain specific and is not perfect. Crossed-out, overwritten, shorthand, cursive, and mixed-script notes can remain unreadable.
- Physically missing content cannot be recovered with certainty. `MODEL_INFERRED` is a proposal, never recovered fact.
- The system does not infer medical diagnoses, medicine names, dosages, dates, identities, legal provisions, case references, or amounts from context in high-risk regions.
- Machine translation is not certified human, legal, or medical translation.
- Automatic region classification is heuristic. Stamps, handwriting, signatures, and diagrams can overlap or be misclassified; debug review exists for this reason.
- Perspective correction handles planar photographs but not full cylindrical/page-curvature dewarping.
- English length changes can force smaller fonts. If wrapping, compact metrics, conservative font reduction, and collision-safe local expansion still cannot fit, the original region is retained; the primary export never creates an overflow appendix or side annotation.
- Scanned text removal uses a bounded ink mask and localized inpainting while excluding long form rules. If the mask looks like a seal, photograph, border, dense graphic, or unreliable background, replacement is rejected and the exact source region is retained.
- PDF visual signatures/stamps are retained, but modifying any digitally signed PDF generally invalidates its cryptographic signature.
- `python-docx` preserves document structures but cannot provide dependable rendered page coordinates or exact Word pagination.
- Highly graphical forms, unusual fonts, equations, dense overlapping notes, and damaged tables may require manual refinement.

## Troubleshooting

**Tesseract not found**: restart Streamlit after installing Tesseract. The app auto-detects standard Windows locations; nonstandard installations can set `DTX_TESSERACT_CMD`.

**`*.traineddata` cannot open**: install the named pack in `.runtime\tessdata` or configure `DTX_TESSDATA_DIRECTORY`. `TESSDATA_PREFIX` remains supported.

**Punjabi/Hindi handwriting says model unavailable**: configure a validated `pa`/`hi` VisionEncoderDecoder checkpoint. Do not point those languages at the English TrOCR model.

**Translation download fails**: run `hf download facebook/nllb-200-distilled-600M`, check disk/network, or configure a compatible local directory.

**CUDA out of memory**: set `DTX_DEVICE=cpu`, reduce `TRANSLATION_BATCH_SIZE`, lower preview/upscale settings, or select smaller compatible models.

**OCR page is worse after enhancement**: select `clean_scan`, reduce upscaling, or compare the original/enhanced/debug tabs. Auto processing guards destructive thresholding but cannot predict every paper/ink combination.

**Handwritten Punjabi page routes as Latin**: select **Expected source language → Punjabi**, enable **Routing debug**, and inspect the visual/OCR/resolved script fields. A Punjabi prior routes probable handwriting to Gurmukhi HTR even when preliminary OCR is ASCII garbage; if no validated Gurmukhi HTR model is configured, the correct safe outcome is `HTR_UNAVAILABLE` with the source crop preserved.

**Mobile photo crops content**: select `clean_scan` or `photocopy` to disable perspective crop for that page, then recapture with the entire page border visible if possible.

**A region remains untranslated in the primary document**: inspect its diagnostic reason. The source reading may have failed validation, HTR may be unavailable, the region may overlap protected graphics, or the English text may not fit without collision. Confirm/correct a credible Punjabi/Hindi source reading in review mode where appropriate; otherwise the original pixels intentionally remain unchanged.

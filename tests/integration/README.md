# Difficult-document integration fixtures

Place private representative fixtures in `tests/fixtures/difficult_documents/`. Fixtures are intentionally not committed because legal and medical scans can contain sensitive data.

Recognized names:

- `punjabi_printed.pdf`
- `hindi_printed.pdf`
- `gurmukhi_handwriting.jpg`
- `devanagari_handwriting.jpg`
- `mixed_medical_form.jpg`
- `low_quality_photocopy.pdf`
- `mobile_photo.jpg`

Run with `python -m pytest -m integration`. Each fixture is skipped when absent. Assertions deliberately check structural/provenance behavior rather than requiring fabricated ground-truth text.


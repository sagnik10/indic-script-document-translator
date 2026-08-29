from streamlit.testing.v1 import AppTest


PRIMARY_OUTPUT_APP = r'''
from types import SimpleNamespace

from translator_app.ui.components import render_downloads

block = SimpleNamespace(metadata={"handwriting_unsupported": True})
result = SimpleNamespace(
    output_bytes=b"primary-document",
    output_filename="source_translated_en.pdf",
    output_mime_type="application/pdf",
    audit_json=b"{}",
    summary=SimpleNamespace(translation_count=0),
    document=SimpleNamespace(blocks=[block]),
)
render_downloads(result)
'''


OPTIONS_APP = r'''
import streamlit as st

from translator_app.config.settings import Settings
from translator_app.ui.components import render_processing_options

options = render_processing_options(Settings())
st.write(f"selected-primary-mode={options.reconstruction_mode.value}")
'''


def test_zero_translation_result_has_no_primary_document_download() -> None:
    app = AppTest.from_string(PRIMARY_OUTPUT_APP, default_timeout=20).run()
    assert not app.exception
    assert any(item.value == "Primary output" for item in app.subheader)
    assert any("becomes available only after" in item.value for item in app.caption)
    assert not any("This file contains only" in item.value for item in app.caption)
    assert any(
        "Optional machine-readable OCR" in item.value for item in app.caption
    )
    assert any("No Punjabi/Hindi source line was translated" in item.value for item in app.error)
    downloads = app.get("download_button")
    assert [item.label for item in downloads] == ["Download OCR/translation audit JSON"]


def test_nonzero_translation_result_exposes_primary_and_diagnostics_downloads() -> None:
    app_source = PRIMARY_OUTPUT_APP.replace(
        "summary=SimpleNamespace(translation_count=0)",
        "summary=SimpleNamespace(translation_count=1)",
    )
    app = AppTest.from_string(app_source, default_timeout=20).run()
    assert not app.exception
    assert [item.label for item in app.get("download_button")] == [
        "Download translated document",
        "Download OCR/translation audit JSON",
    ]


def test_processing_options_expose_only_format_preserving_primary_mode() -> None:
    app = AppTest.from_string(OPTIONS_APP, default_timeout=20).run()
    assert not app.exception
    assert not any(item.label == "Reconstruction mode" for item in app.radio)
    assert any(
        "selected-primary-mode=clean_rebuild" in item.value for item in app.markdown
    )

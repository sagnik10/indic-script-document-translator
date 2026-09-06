"""Streamlit application with optional pre-translation OCR review."""

from __future__ import annotations

import hashlib
import logging

import streamlit as st

from ..config.settings import get_settings
from ..core.source_validation import (
    SUPPORTED_TRANSLATION_LANGUAGES,
    normalize_language,
)
from ..exceptions import DocumentTranslatorError
from ..pipeline import DocumentTranslationPipeline
from ..schemas import ProcessingStatus
from .components import (
    create_progress_callback,
    render_analysis_previews,
    render_diagnostic_previews,
    render_downloads,
    render_limitations,
    render_previews,
    render_processing_options,
    render_review_editor,
    render_routing_debug,
    render_summary,
    render_uncertainty_table,
    render_upload_metadata,
)

LOGGER = logging.getLogger(__name__)


def _unresolved_htr_block_ids(analysis: object) -> set[str]:
    document = getattr(analysis, "document", None)
    if not document:
        return set()
    return {
        block.block_id
        for block in document.blocks
        if (
            (
                ProcessingStatus.HTR_UNAVAILABLE in block.processing_statuses
                or bool(block.metadata.get("htr_unavailable"))
            )
            and not bool(block.metadata.get("human_reviewed"))
        )
    }


def _has_unresolved_htr(analysis: object) -> bool:
    return bool(_unresolved_htr_block_ids(analysis))


def _requires_review_pause(analysis: object) -> bool:
    options = getattr(analysis, "options", None)
    return bool(getattr(options, "review_before_render", False) or _has_unresolved_htr(analysis))


def _requires_translated_output(analysis: object) -> bool:
    """Return whether finalization must produce at least one real Indic translation.

    Unresolved handwriting and validated Punjabi/Hindi source both describe a
    translation request.  In either case an unchanged source-template PDF must
    not be exposed under a translated filename.
    """

    if _has_unresolved_htr(analysis):
        return True
    document = getattr(analysis, "document", None)
    if not document:
        return False
    return any(
        bool(block.source_validated)
        and normalize_language(block.detected_language)
        in SUPPORTED_TRANSLATION_LANGUAGES
        for block in document.blocks
    )


def _validated_indic_block_count(analysis: object) -> int:
    """Count source-validated Punjabi/Hindi blocks ready for translation."""

    document = getattr(analysis, "document", None)
    if not document:
        return 0
    return sum(
        bool(block.source_validated)
        and normalize_language(block.detected_language)
        in SUPPORTED_TRANSLATION_LANGUAGES
        for block in document.blocks
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> DocumentTranslationPipeline:
    return DocumentTranslationPipeline(get_settings())


def _progress_widgets():
    progress_bar = st.progress(0.0, text="Preparing local pipeline")
    status = st.status("Preparing", state="running", expanded=True)
    stage_area = status.empty()
    return status, create_progress_callback(progress_bar, status, stage_area)


def _show_error(exc: Exception, status: object) -> None:
    settings = get_settings()
    LOGGER.exception("Document processing failed")
    status.update(label="Processing failed", state="error", expanded=True)
    if isinstance(exc, DocumentTranslatorError):
        st.error(exc.user_message)
    elif isinstance(exc, PermissionError):
        st.error(
            "A local file permission prevented processing. Check configured temporary/output directory access."
        )
    else:
        st.error(
            "An unexpected local processing error occurred. Technical details were written to the log."
        )
    if settings.debug:
        st.exception(exc)


def run() -> None:
    st.set_page_config(
        page_title="Difficult Document Translator",
        page_icon=":material/translate:",
        layout="wide",
    )
    st.title("Punjabi & Hindi Difficult-Document Translator")
    st.write(
        "Local, region-aware OCR/HTR, conservative uncertainty review, Indic-to-English translation, "
        "and format-preserving PDF/DOCX reconstruction for scans, photocopies, forms, and photographs."
    )
    settings = get_settings()
    options = render_processing_options(settings)
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, PNG, JPG, or JPEG",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help=f"Maximum configured size: {settings.max_upload_size / (1024 * 1024):.0f} MB",
    )
    render_limitations()
    if uploaded_file is None:
        st.info("Choose a source document to begin.")
        return
    render_upload_metadata(uploaded_file)
    file_bytes = uploaded_file.getvalue()
    upload_key = f"{uploaded_file.name}:{len(file_bytes)}:{hashlib.sha256(file_bytes).hexdigest()[:12]}"
    if st.session_state.get("active_upload_key") != upload_key:
        st.session_state["active_upload_key"] = upload_key
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("latest_result", None)
        for key in list(st.session_state):
            if str(key).startswith(
                (
                    "htr_text_",
                    "htr_confirm_",
                    "missing_span_text_",
                    "missing_span_confirm_",
                    "uncertain_ocr_editor",
                    "routing_crops_",
                )
            ):
                st.session_state.pop(key, None)

    if st.button("Analyze document", type="primary", width="stretch"):
        # A fresh analysis of the same upload must not be hidden by a stale output
        # retained from an earlier run in this browser session.
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("latest_result", None)
        status, callback = _progress_widgets()
        try:
            analysis = get_pipeline().analyze(
                uploaded_file.name, file_bytes, options, callback
            )
            st.session_state["analysis_result"] = analysis
            if _has_unresolved_htr(analysis):
                # A missing language-capable HTR provider cannot be bypassed by
                # disabling optional low-confidence review.  The user must see
                # the source crops and confirm at least one credible line before
                # a translated-document export can be created. Other lines may
                # remain visually preserved in a partial translation.
                analysis.options.review_before_render = True
                analysis.document.metadata["mandatory_htr_review"] = True
                st.warning(
                    "Automatic Punjabi/Hindi handwriting transcription is unavailable for one "
                    "or more lines. Analysis has finished; processing is paused for manual "
                    "source-language review below."
                )
            if not _requires_review_pause(analysis):
                st.session_state["latest_result"] = get_pipeline().finalize(
                    analysis,
                    callback,
                    require_translation=_requires_translated_output(analysis),
                )
            else:
                st.success(
                    "Analysis complete. Scroll down, transcribe and confirm at least one readable "
                    "Punjabi/Hindi line, then select Translate confirmed source lines and render."
                )
        except Exception as exc:
            _show_error(exc, status)
            return

    analysis = st.session_state.get("analysis_result")
    result = st.session_state.get("latest_result")
    if analysis is not None and result is None:
        render_analysis_previews(analysis)
        if analysis.options.routing_debug:
            render_routing_debug(analysis.document)
        edits, submitted = render_review_editor(analysis)
        if submitted:
            unresolved_before = _unresolved_htr_block_ids(analysis)
            if unresolved_before and not edits:
                st.warning(
                    "No source transcription was confirmed. Enter the Punjabi/Hindi text for "
                    "at least one readable crop, select its confirmation checkbox, and submit again."
                )
                return
            try:
                get_pipeline().apply_review_edits(analysis, edits)
            except Exception as exc:
                status, _callback = _progress_widgets()
                _show_error(exc, status)
                return
            unresolved_after = _unresolved_htr_block_ids(analysis)
            rejected_ids = unresolved_after.intersection(edits)
            if rejected_ids:
                st.warning(
                    f"{len(rejected_ids)} submitted source transcription(s) did not pass "
                    "Punjabi/Hindi script and source-quality validation. Correct them above, "
                    "or leave them unconfirmed so their original pixels remain unchanged."
                )
            preserved_count = len(unresolved_after.difference(rejected_ids))
            if preserved_count and edits:
                st.info(
                    f"{preserved_count} unconfirmed handwritten line(s) will remain visually "
                    "unchanged. Only confirmed, validated lines will be translated."
                )
            require_translation = bool(
                unresolved_before
                or unresolved_after
                or _requires_translated_output(analysis)
            )
            if require_translation and _validated_indic_block_count(analysis) == 0:
                st.warning(
                    "No confirmed transcription passed Punjabi/Hindi source validation. "
                    "Correct at least one source-language line and submit again. Translation "
                    "and rendering were not started."
                )
                return
            status, callback = _progress_widgets()
            try:
                result = get_pipeline().finalize(
                    analysis,
                    callback,
                    require_translation=require_translation,
                )
                st.session_state["latest_result"] = result
                st.success("The English document is ready for verification and download.")
            except Exception as exc:
                _show_error(exc, status)
                return
    if result is not None:
        if result.summary.translation_count > 0:
            st.header("Translated document")
            render_previews(result)
        else:
            # Defensive handling for stale/pre-guard session results: never label
            # or preview an unchanged source page as a translated document.
            st.header("Translation not generated")
        render_downloads(result)
        with st.expander(
            "Diagnostics and processing details",
            expanded=False,
            icon=":material/troubleshoot:",
        ):
            render_diagnostic_previews(result)
            if bool(result.document.metadata.get("routing_debug")):
                render_routing_debug(result.document)
            render_summary(result)
            render_uncertainty_table(result)
            st.subheader("Local model status")
            st.json(get_pipeline().models.status())

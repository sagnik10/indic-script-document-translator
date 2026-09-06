"""Composable Streamlit controls, review tables, previews, summaries, and downloads."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..config.settings import Settings
from ..schemas import (
    AnalysisResult,
    DocumentModel,
    ProcessingStatus,
    ProcessingOptions,
    ProcessingResult,
    ProcessingStage,
    ReconstructionStatus,
    ReconstructionType,
    ReconstructionMode,
)


LANGUAGE_PACKS = {
    "English": "eng",
    "Punjabi (Gurmukhi)": "pan",
    "Hindi (Devanagari)": "hin",
    "Bengali": "ben",
    "Marathi": "mar",
    "Gujarati": "guj",
    "Tamil": "tam",
    "Telugu": "tel",
    "Kannada": "kan",
    "Malayalam": "mal",
    "Odia": "ori",
}

PROFILES = [
    "auto",
    "clean_scan",
    "photocopy",
    "mobile_photo",
    "faded_document",
    "handwriting_heavy",
]


def _parse_glossary(text: str) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        if source.strip() and target.strip():
            glossary[source.strip()] = target.strip()
    return glossary


def render_upload_metadata(uploaded_file: object) -> None:
    size = getattr(uploaded_file, "size", 0)
    columns = st.columns(3)
    columns[0].metric("Filename", getattr(uploaded_file, "name", "Unknown"))
    columns[1].metric(
        "File type", Path(getattr(uploaded_file, "name", "")).suffix.upper() or "Unknown"
    )
    columns[2].metric("Size", f"{size / (1024 * 1024):.2f} MB")


def render_processing_options(settings: Settings) -> ProcessingOptions:
    with st.sidebar:
        st.header("Difficult-document controls")
        selected_names = st.multiselect(
            "Printed OCR languages",
            options=list(LANGUAGE_PACKS),
            default=[
                name
                for name, code in LANGUAGE_PACKS.items()
                if code in (settings.ocr_languages or [])
            ],
            help="Punjabi, Hindi, and English are prioritized. Only installed Tesseract packs run.",
        )
        profile = st.selectbox(
            "Preprocessing profile",
            PROFILES,
            index=PROFILES.index(settings.preprocessing_profile)
            if settings.preprocessing_profile in PROFILES
            else 0,
            help="Auto evaluates blur, contrast, noise, shadows, skew, and page borders first.",
        )
        upscale = st.select_slider(
            "OCR upscaling",
            options=[1.0, 2.0, 3.0, 4.0],
            value=float(settings.ocr_upscale_factor),
            help="Applied only when image resolution or sharpness warrants it.",
        )
        force_ocr = st.checkbox("Force OCR on native PDF pages", value=False)
        enable_printed = st.checkbox("Printed-text OCR", value=True)
        enable_handwriting = st.checkbox(
            "Handwriting recognition",
            value=True,
            help="Requires a configured language-specific HTR model. Unreadable handwriting stays visible.",
        )
        expected_language_label = st.segmented_control(
            "Expected source language",
            ["Auto", "Punjabi", "Hindi", "Punjabi + Hindi"],
            default={
                "auto": "Auto",
                "pa": "Punjabi",
                "hi": "Hindi",
                "pa+hi": "Punjabi + Hindi",
            }.get(settings.expected_source_language, "Auto"),
            help=(
                "A routing prior for difficult pages. Recognition output must still pass Unicode "
                "script and source-quality validation before translation."
            ),
        )
        expected_language = {
            "Auto": "auto",
            "Punjabi": "pa",
            "Hindi": "hi",
            "Punjabi + Hindi": "pa+hi",
        }[expected_language_label or "Auto"]
        handwriting_hint = expected_language if expected_language in {"pa", "hi"} else "auto"
        preserve_handwriting = st.checkbox(
            "Preserve unreadable handwriting as image",
            value=settings.preserve_unreadable_handwriting_as_image,
        )
        st.caption(
            "Primary output uses strict format-preserving reconstruction. Unsafe or unreadable "
            "regions remain exactly as source content instead of being moved to a transcript."
        )
        domain = st.selectbox(
            "Document risk profile",
            ["auto", "medical", "legal", "government", "general"],
            help="Medical/legal/government modes prohibit contextual model guesses in high-risk regions.",
        )
        enable_reconstruction = st.checkbox("Conservative OCR reconstruction", value=True)
        ocr_threshold = st.slider(
            "Low printed-OCR confidence",
            0.0,
            1.0,
            float(settings.ocr_low_confidence_threshold),
            0.01,
        )
        handwriting_threshold = st.slider(
            "Low handwriting confidence",
            0.0,
            1.0,
            float(settings.handwriting_confidence_threshold),
            0.01,
        )
        reconstruction_threshold = st.slider(
            "Legacy deterministic-correction threshold",
            0.0,
            1.0,
            float(settings.reconstruction_accept_threshold),
            0.01,
            disabled=not enable_reconstruction,
        )
        auto_reconstruct_threshold = st.slider(
            "Auto-accept missing-span confidence",
            0.0,
            1.0,
            float(settings.auto_reconstruct_threshold),
            0.01,
            disabled=not enable_reconstruction,
            help="Candidates above this threshold are still marked MODEL_INFERRED.",
        )
        review_reconstruct_threshold = st.slider(
            "Review missing-span confidence",
            0.0,
            float(auto_reconstruct_threshold),
            min(float(settings.review_reconstruct_threshold), float(auto_reconstruct_threshold)),
            0.01,
            disabled=not enable_reconstruction,
            help="Candidates between review and auto thresholds require explicit confirmation.",
        )
        min_context_quality = st.slider(
            "Minimum source-context quality",
            0.0,
            1.0,
            float(settings.min_context_quality),
            0.01,
            disabled=not enable_reconstruction,
        )
        review_before_render = st.checkbox(
            "Pause for uncertain-OCR review",
            value=True,
            help="Allows explicit corrections before translation and final rendering.",
        )
        debug_boxes = st.checkbox("Debug OCR", value=False)
        routing_debug = st.toggle(
            "Routing debug",
            value=False,
            help="Shows visual-first page/region routing evidence without changing processing.",
        )
        with st.expander("Terminology and protected terms"):
            protected_text = st.text_area(
                "Protected terms (one per line)",
                placeholder="PGIMER\nCivil Hospital\nAmoxicillin",
            )
            glossary_text = st.text_area(
                "Glossary entries (source=approved English)",
                placeholder="ਤਹਿਸੀਲ=Tehsil\nएफआईआर=FIR",
            )
        st.caption("Local processing is the default. Full document text is not written to logs.")
    return ProcessingOptions(
        target_language="en",
        force_ocr=force_ocr,
        enable_preprocessing=True,
        enable_reconstruction=enable_reconstruction,
        ocr_languages=[LANGUAGE_PACKS[name] for name in selected_names] or ["eng"],
        ocr_low_confidence_threshold=ocr_threshold,
        handwriting_confidence_threshold=handwriting_threshold,
        reconstruction_accept_threshold=reconstruction_threshold,
        auto_reconstruct_threshold=auto_reconstruct_threshold,
        review_reconstruct_threshold=review_reconstruct_threshold,
        min_context_quality=min_context_quality,
        preprocessing_profile=profile,
        ocr_upscale_factor=upscale,
        enable_printed_ocr=enable_printed,
        enable_handwriting_ocr=enable_handwriting,
        handwriting_language_hint=handwriting_hint,
        expected_source_language=expected_language,
        routing_debug=routing_debug,
        preserve_unreadable_handwriting_as_image=preserve_handwriting,
        reconstruction_mode=ReconstructionMode.CLEAN_REBUILD,
        review_before_render=review_before_render,
        debug_bounding_boxes=debug_boxes,
        document_domain=domain,
        protected_terms=[line.strip() for line in protected_text.splitlines() if line.strip()],
        glossary=_parse_glossary(glossary_text),
    )


def create_progress_callback(progress_bar: object, status: object, stage_area: object):
    completed: list[str] = []

    def callback(stage: ProcessingStage, progress: float, detail: str) -> None:
        if stage == ProcessingStage.REVIEW:
            progress_bar.progress(
                1.0,
                text="OCR analysis complete — source review required",
            )
        else:
            progress_bar.progress(progress, text=detail)
        if stage.value not in completed:
            completed.append(stage.value)
        stage_area.markdown("  \n".join(f"✓ {name}" for name in completed))
        if stage == ProcessingStage.COMPLETE:
            status.update(label="Processing complete", state="complete", expanded=False)
        elif stage == ProcessingStage.REVIEW:
            status.update(
                label="OCR analysis complete — review below",
                state="complete",
                expanded=False,
            )
        else:
            status.update(label=stage.value, state="running", expanded=True)

    return callback


def _image_navigator(title: str, images: list[bytes], key: str) -> None:
    st.subheader(title)
    if not images:
        st.info("No raster preview is available for this format/page.")
        return
    page = st.number_input(
        f"{title} page",
        min_value=1,
        max_value=len(images),
        value=1,
        step=1,
        key=key,
    )
    st.image(images[int(page) - 1], width="stretch")


def render_analysis_previews(analysis: AnalysisResult) -> None:
    tabs = st.tabs(["Original", "Enhanced OCR image", "Review overlay"])
    with tabs[0]:
        _image_navigator("Original page", analysis.source_preview_images, "analysis_source_page")
    with tabs[1]:
        _image_navigator(
            "Enhanced page", analysis.enhanced_preview_images, "analysis_enhanced_page"
        )
    with tabs[2]:
        st.caption(
            "Green: high confidence · orange: low confidence · purple: reconstructed/handwriting · red: unreadable/flagged."
        )
        if analysis.options.debug_bounding_boxes:
            _image_navigator("Detected regions", analysis.debug_preview_images, "analysis_debug_page")
        else:
            st.info("Enable Debug OCR in the sidebar to display bounding boxes.")


def render_routing_debug(document: DocumentModel) -> None:
    """Display visual, OCR-derived, and resolved routing as separate evidence."""
    with st.expander("Routing debug", expanded=True, icon=":material/route:"):
        st.caption(
            "Visual classification runs before OCR. OCR-derived script is counted only when it "
            "contains meaningful linguistic evidence. Heuristic visual script confidence is not a guarantee."
        )
        for page in document.pages:
            st.markdown(f"#### Page {page.page_number}")
            first = st.columns(4)
            first[0].metric("Expected language", page.expected_language_prior)
            first[1].metric(
                "Visual script",
                page.visual_page_script.value,
                f"{page.visual_page_script_confidence:.1%}",
            )
            first[2].metric(
                "OCR-derived script",
                page.ocr_page_script.value,
                f"{page.ocr_page_script_confidence:.1%}",
            )
            first[3].metric(
                "Resolved script",
                page.resolved_page_script.value,
                f"{page.resolved_page_script_confidence:.1%}",
            )
            second = st.columns(4)
            second[0].metric("Page handwriting", f"{page.handwriting_probability:.1%}")
            second[1].metric("Page type", page.page_visual_type.value)
            second[2].metric(
                "Detected text lines",
                int(page.metadata.get("detected_text_line_count", 0)),
            )
            second[3].metric(
                "Rejected/noise",
                int(page.metadata.get("rejected_noise_regions", 0)),
            )
            third = st.columns(3)
            third[0].metric(
                "Punjabi HTR routes", int(page.metadata.get("punjabi_htr_routes", 0))
            )
            third[1].metric(
                "Hindi HTR routes", int(page.metadata.get("hindi_htr_routes", 0))
            )
            third[2].metric(
                "Printed OCR routes", int(page.metadata.get("printed_ocr_routes", 0))
            )
            st.info(f"Resolution reason: {page.script_resolution_reason}")
            region_rows = []
            for region in sorted(page.regions, key=lambda item: item.reading_order):
                region_blocks = [
                    block
                    for block in page.blocks
                    if str(block.metadata.get("region_id", "")) == region.region_id
                ]
                recognized = " ".join(
                    (block.normalized_text or block.source_text).strip()
                    for block in region_blocks
                    if (block.normalized_text or block.source_text).strip()
                )
                validation = "; ".join(
                    dict.fromkeys(block.validation_reason for block in region_blocks)
                )
                region_rows.append(
                    {
                        "Region": region.region_id[:10],
                        "Type": region.region_type.value,
                        "Visual script": region.visual_script_candidate.value,
                        "Visual confidence": region.visual_script_confidence,
                        "Resolved script": region.resolved_script.value,
                        "Recognition engine": region.selected_recognition_engine,
                        "Recognition result": recognized or "[no source text]",
                        "Unicode script": region.recognized_unicode_script.value,
                        "Linguistic evidence": region.linguistic_evidence_score,
                        "Source validation": validation or "not validated",
                        "Reason": region.script_resolution_reason,
                    }
                )
            if region_rows:
                st.dataframe(
                    region_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Visual confidence": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "Linguistic evidence": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="percent"
                        ),
                    },
                )
            show_crops = st.toggle(
                f"Show page {page.page_number} routing crops",
                value=False,
                key=f"routing_crops_{page.page_number}",
            )
            if show_crops:
                for region in sorted(page.regions, key=lambda item: item.reading_order):
                    block = next(
                        (
                            item
                            for item in page.blocks
                            if str(item.metadata.get("region_id", "")) == region.region_id
                        ),
                        None,
                    )
                    if block is None or block.review_image_bytes is None:
                        continue
                    with st.container(border=True):
                        image_column, route_column = st.columns(
                            [1, 2], vertical_alignment="center"
                        )
                        image_column.image(block.review_image_bytes, width="stretch")
                        route_column.write(
                            f"{region.region_type.value} → {region.visual_script_candidate.value} "
                            f"→ {region.selected_recognition_engine} → "
                            f"{region.recognized_unicode_script.value}"
                        )
                        route_column.caption(
                            f"Validation: {block.validation_reason} · "
                            f"Unicode ratio: {block.metadata.get('unicode_script_ratio', 0.0):.1%}"
                        )


def render_review_editor(analysis: AnalysisResult) -> tuple[dict[str, str], bool]:
    uncertain = [block for block in analysis.document.blocks if block.is_uncertain]
    st.subheader(f"Uncertain OCR review ({len(uncertain)} blocks)")
    if not uncertain:
        st.success("No uncertain OCR blocks require review.")
        submitted = st.button(
            "Translate and render validated source",
            type="primary",
            width="stretch",
        )
        return {}, submitted

    automatic_reconstructions = [
        block
        for block in uncertain
        if block.reconstruction_status == ReconstructionStatus.AUTO_ACCEPTED
        and block.reconstruction_type == ReconstructionType.MODEL_INFERRED
    ]
    reconstruction_review = [
        block
        for block in uncertain
        if block.reconstruction_status == ReconstructionStatus.CANDIDATE_REVIEW
    ]
    dedicated_ids = {
        block.block_id for block in automatic_reconstructions + reconstruction_review
    }
    handwriting = [
        block for block in uncertain if block.is_handwritten and block.block_id not in dedicated_ids
    ]
    unsupported_handwriting = [
        block
        for block in handwriting
        if ProcessingStatus.HTR_UNAVAILABLE in block.processing_statuses
        or bool(block.metadata.get("htr_unavailable"))
    ]
    other = [
        block for block in uncertain if not block.is_handwritten and block.block_id not in dedicated_ids
    ]
    edits: dict[str, str] = {}
    with st.form("source_review_form"):
        if unsupported_handwriting:
            source_languages = sorted(
                {
                    "Punjabi"
                    if block.detected_language == "pa"
                    else "Hindi"
                    if block.detected_language == "hi"
                    else "source-language"
                    for block in unsupported_handwriting
                }
            )
            st.warning(
                f"Automatic {' / '.join(source_languages)} handwriting transcription is unavailable "
                f"for {len(unsupported_handwriting)} line(s). To translate them, type the exact "
                "source-language text visible in each crop and confirm it. The app will validate "
                "the script before translation; unconfirmed handwriting remains unchanged."
            )
        if automatic_reconstructions:
            st.markdown("#### Automatically reconstructed source spans")
            st.caption(
                "These source-language candidates exceeded the auto-accept threshold. They remain "
                "marked MODEL_INFERRED in the audit and should be visually checked."
            )
            for block in automatic_reconstructions:
                with st.container(border=True):
                    image_column, detail_column = st.columns([1, 2], vertical_alignment="center")
                    with image_column:
                        if block.review_image_bytes:
                            st.image(block.review_image_bytes, width="stretch")
                        else:
                            st.info("Verify this span against the source-page preview.")
                    with detail_column:
                        st.success(
                            f"MODEL_INFERRED · confidence {(block.reconstruction_confidence or 0.0):.1%}"
                        )
                        st.write(f"Source OCR: {block.normalized_text or block.source_text}")
                        st.write(f"Inferred source span: {block.reconstruction_candidate}")
                        st.write(f"Reconstructed source: {block.reconstructed_text}")
                        st.caption(
                            f"Method: {block.reconstruction_method or 'unknown'} · "
                            f"script: {block.script.value} · page {block.page_number}"
                        )

        if reconstruction_review:
            st.markdown("#### Missing source spans requiring confirmation")
            st.caption(
                "Only confirm a candidate after comparing the crop and surrounding source-language "
                "context. Text outside the bounded missing span cannot be changed here."
            )
            for block in reconstruction_review:
                with st.container(border=True):
                    image_column, detail_column = st.columns([1, 2], vertical_alignment="center")
                    with image_column:
                        if block.review_image_bytes:
                            st.image(
                                block.review_image_bytes,
                                caption=f"Page {block.page_number} · missing source span",
                                width="stretch",
                            )
                        else:
                            st.info("Verify this span against the source-page preview.")
                    with detail_column:
                        confidence = block.reconstruction_confidence or 0.0
                        st.warning(f"Source reconstruction requires review · {confidence:.1%}")
                        previous = str(block.metadata.get("missing_span_previous_text", "")).strip()
                        following = str(block.metadata.get("missing_span_next_text", "")).strip()
                        if previous:
                            st.caption(f"Previous source line: {previous}")
                        st.write(f"OCR source line: {block.normalized_text or block.source_text}")
                        if following:
                            st.caption(f"Next source line: {following}")
                        st.write(f"Proposed {block.script.value} span: {block.reconstruction_candidate}")
                        proposed = str(
                            block.metadata.get("proposed_reconstructed_source_text", "")
                        )
                        reviewed = st.text_area(
                            "Confirmed source line",
                            value=proposed,
                            key=f"missing_span_text_{block.block_id}",
                            help="Edit only the proposed Punjabi/Hindi missing span, not the surrounding text.",
                        )
                        confirmed = st.checkbox(
                            "I confirm this bounded source-language reconstruction",
                            key=f"missing_span_confirm_{block.block_id}",
                        )
                        if confirmed and reviewed.strip():
                            edits[block.block_id] = reviewed

        if handwriting:
            st.markdown("#### Handwritten source-line review")
            st.caption(
                "Compare each crop with the Punjabi/Hindi source. Select a line only after the "
                "source-language transcription is credible; translation happens after submission. "
                "You may confirm only the lines you can read, and all remaining lines will retain "
                "their original pixels. At least one confirmed line must pass validation and "
                "translate before a primary document can be exported."
            )
            for block in handwriting:
                with st.container(border=True):
                    image_column, text_column = st.columns([1, 2], vertical_alignment="center")
                    with image_column:
                        if block.review_image_bytes:
                            st.image(
                                block.review_image_bytes,
                                caption=f"Page {block.page_number} · source line",
                                width="stretch",
                            )
                        else:
                            st.info("The source-line crop is unavailable; verify against the page preview.")
                    with text_column:
                        if block in unsupported_handwriting:
                            st.caption(
                                "HTR unavailable - manual source transcription is required for this "
                                "line to be translated. Do not enter English here."
                            )
                        confidence = (
                            f"{block.ocr_confidence:.1%}"
                            if block.ocr_confidence is not None
                            else "not reported"
                        )
                        st.caption(
                            f"Page {block.page_number} · block {block.block_id[:10]} · "
                            f"{block.script.value} · HTR confidence {confidence} · {block.ocr_engine}"
                        )
                        candidate = block.effective_source_text
                        if candidate.startswith("[") and candidate.endswith("]"):
                            candidate = ""
                        reviewed = st.text_area(
                            "Recognized Punjabi/Hindi source text",
                            value=candidate,
                            placeholder="Type the source-language transcription exactly as visible",
                            key=f"htr_text_{block.block_id}",
                            help="Do not translate into English here. Enter only the visible source text.",
                        )
                        confirmed = st.checkbox(
                            "I confirm this source transcription for validation and translation",
                            key=f"htr_confirm_{block.block_id}",
                        )
                        if confirmed and reviewed.strip():
                            edits[block.block_id] = reviewed

        if other:
            st.markdown("#### Other uncertain OCR")
            rows = [
                {
                    "Use edit": False,
                    "Block ID": block.block_id,
                    "Page": block.page_number,
                    "Region": block.region_type.value,
                    "Engine": block.ocr_engine,
                    "Confidence": block.ocr_confidence,
                    "Text quality": block.text_quality,
                    "Language": block.detected_language,
                    "Reason": block.validation_reason,
                    "State": block.reconstruction_type.value,
                    "Raw OCR": block.source_text,
                    "Candidate / reviewed reading": block.effective_source_text,
                }
                for block in other
            ]
            edited = st.data_editor(
                rows,
                width="stretch",
                hide_index=True,
                disabled=[
                    "Block ID",
                    "Page",
                    "Region",
                    "Engine",
                    "Confidence",
                    "Text quality",
                    "Language",
                    "Reason",
                    "State",
                    "Raw OCR",
                ],
                column_config={
                    "Use edit": st.column_config.CheckboxColumn(
                        "Use edit", help="Only checked source-language edits are applied."
                    ),
                    "Candidate / reviewed reading": st.column_config.TextColumn(width="large"),
                },
                key="uncertain_ocr_editor",
            )
            records = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
            edits.update(
                {
                    str(row["Block ID"]): str(row["Candidate / reviewed reading"])
                    for row in records
                    if row.get("Use edit")
                    and str(row.get("Candidate / reviewed reading", "")).strip()
                }
            )
        submitted = st.form_submit_button(
            "Translate confirmed source lines and render",
            type="primary",
            width="stretch",
        )
    return edits, submitted


def _text_preview(result: ProcessingResult, source: bool) -> None:
    rows = [
        {
            "Page": block.page_number,
            "Region": block.region_type.value,
            "Language": block.detected_language,
            "Confidence": block.ocr_confidence,
            "Text": block.source_text if source else block.output_text,
            "Status": block.reconstruction_type.value
            if source
            else block.translation_status.value,
        }
        for block in result.document.blocks[:200]
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def render_previews(result: ProcessingResult) -> None:
    """Render only the reconstructed translated document in the primary preview."""
    if result.output_preview_images:
        _image_navigator(
            "Translated document preview",
            result.output_preview_images,
            "result_output_page",
        )
    else:
        st.subheader("Translated document structure")
        _text_preview(result, False)


def render_diagnostic_previews(result: ProcessingResult) -> None:
    """Render source and OCR imagery only inside the optional diagnostics area."""
    if result.source_preview_images:
        _image_navigator("Original source", result.source_preview_images, "result_source_page")
    else:
        st.subheader("Original source structure")
        _text_preview(result, True)
    if result.enhanced_preview_images:
        _image_navigator(
            "Enhanced OCR input",
            result.enhanced_preview_images,
            "result_enhanced_page",
        )
    if result.debug_preview_images:
        _image_navigator(
            "Debug OCR overlay",
            result.debug_preview_images,
            "result_debug_page",
        )


def render_uncertainty_table(result: ProcessingResult) -> None:
    rows = [
        {
            "Page": block.page_number,
            "Block": block.block_id[:10],
            "Region type": block.region_type.value,
            "Source text": block.effective_source_text,
            "Script": block.script.value,
            "Language": block.detected_language,
            "OCR confidence": block.ocr_confidence,
            "Text quality": block.text_quality,
            "Reconstruction": block.reconstruction_status.value,
            "Candidate": block.reconstruction_candidate,
            "Translation status": block.translation_status.value,
            "Reason": block.validation_reason,
        }
        for block in result.document.blocks
        if block.is_uncertain
    ]
    if rows:
        with st.expander(f"Uncertain source regions ({len(rows)})", expanded=True):
            st.dataframe(rows, width="stretch", hide_index=True)


def render_summary(result: ProcessingResult) -> None:
    summary = result.summary
    st.subheader("Processing summary")
    if result.document.metadata.get("ocr_quality_abort_page_count", 0):
        st.error(
            "OCR quality was too low for reliable automatic translation on one or more pages. "
            "Those source regions were preserved; review the uncertain-region table and audit JSON."
        )
    first = st.columns(6)
    first[0].metric("Pages", summary.page_count)
    first[1].metric("Regions", summary.region_count)
    first[2].metric("Text blocks", summary.text_block_count)
    first[3].metric("OCR blocks", summary.ocr_block_count)
    first[4].metric("Handwritten regions", summary.handwritten_region_count)
    first[5].metric("Duration", f"{summary.processing_duration_seconds:.1f}s")
    second = st.columns(6)
    second[0].metric("Dominant script", ", ".join(summary.dominant_scripts) or "Unknown")
    second[1].metric("Printed regions", summary.printed_region_count)
    second[2].metric("HTR-recognized lines", summary.htr_recognized_line_count)
    second[3].metric("Manually reviewed lines", summary.manually_reviewed_line_count)
    second[4].metric("Rejected handwriting", summary.rejected_handwriting_line_count)
    second[5].metric("Low-confidence", summary.low_confidence_ocr_count)
    third = st.columns(6)
    third[0].metric(
        "Validated PA / HI",
        f"{summary.validated_punjabi_line_count} / {summary.validated_hindi_line_count}",
    )
    third[1].metric("Unreadable", summary.unreadable_block_count)
    third[2].metric(
        "Translated / skipped",
        f"{summary.translation_count} / {summary.translation_skipped_count}",
    )
    third[3].metric("Layout overflow", summary.layout_overflow_count)
    third[4].metric("HTR blocks", summary.handwriting_block_count)
    third[5].metric("Uncertain blocks", summary.uncertain_block_count)
    st.caption(
        f"Profiles: {', '.join(summary.preprocessing_profiles) or 'native'} · "
        f"Input: {summary.filename} · Output: {summary.output_filename}"
    )
    st.caption(
        f"Missing spans: {summary.missing_span_count} · "
        f"auto reconstructed: {summary.auto_reconstructed_span_count} · "
        f"awaiting review: {summary.review_reconstruction_count} · "
        f"manually confirmed: {summary.manually_confirmed_count} · "
        f"unresolved: {summary.unresolved_missing_span_count}"
    )
    st.caption(
        f"Page types: {', '.join(summary.page_visual_types) or 'unknown'}; "
        f"Punjabi HTR routes: {summary.punjabi_htr_route_count}; "
        f"Hindi HTR routes: {summary.hindi_htr_route_count}; "
        f"printed OCR routes: {summary.printed_ocr_route_count}; "
        f"rejected/noise regions: {summary.rejected_noise_region_count}"
    )
    page_fidelity = [
        (page.page_number, page.metadata.get("format_fidelity_score"))
        for page in result.document.pages
        if isinstance(page.metadata.get("format_fidelity_score"), (int, float))
    ]
    block_fidelity = [
        {
            "Page": block.page_number,
            "Block": block.block_id[:10],
            "Fidelity": float(block.metadata["format_fidelity_score"]),
            "Replacement": block.metadata.get(
                "primary_output_replacement_reason", "rendered"
            ),
        }
        for block in result.document.blocks
        if isinstance(block.metadata.get("format_fidelity_score"), (int, float))
    ]
    if page_fidelity or block_fidelity:
        st.subheader("Format fidelity")
        if page_fidelity:
            st.caption(
                " · ".join(
                    f"Page {page}: {float(score):.1%}"
                    for page, score in page_fidelity
                )
            )
        if block_fidelity:
            st.dataframe(
                block_fidelity,
                width="stretch",
                hide_index=True,
                column_config={
                    "Fidelity": st.column_config.ProgressColumn(
                        "Fidelity", min_value=0.0, max_value=1.0, format="percent"
                    )
                },
            )
    if summary.warnings:
        with st.expander(f"Warnings ({len(summary.warnings)})", expanded=True):
            for warning in summary.warnings:
                st.warning(warning)


def render_downloads(result: ProcessingResult) -> None:
    st.subheader("Primary output")
    if result.summary.translation_count <= 0:
        st.caption(
            "A primary document becomes available only after at least one validated "
            "Punjabi/Hindi source line has been translated."
        )
        st.error(
            "No Punjabi/Hindi source line was translated, so no primary translated-document "
            "download is available. Confirm at least one credible source-language transcription "
            "in the review step and try again; unconfirmed regions will remain unchanged."
        )
    else:
        st.caption(
            "This file contains only the format-preserved translated document. Diagnostics are "
            "not appended to it."
        )
        st.download_button(
            "Download translated document",
            data=result.output_bytes,
            file_name=result.output_filename,
            mime=result.output_mime_type,
            type="primary",
            width="stretch",
        )
    with st.expander("Download diagnostics", expanded=False, icon=":material/diagnosis:"):
        st.caption(
            "Optional machine-readable OCR, routing, reconstruction, translation, and layout "
            "provenance. This JSON is separate from the primary document."
        )
        st.download_button(
            "Download OCR/translation audit JSON",
            data=result.audit_json,
            file_name=f"{Path(result.output_filename).stem}_audit.json",
            mime="application/json",
            width="stretch",
        )


def render_limitations() -> None:
    with st.expander("Accuracy, medical/legal safety, privacy, and layout limitations"):
        st.markdown(
            """
- OCR and handwriting recognition are probabilistic. Punjabi/Hindi handwriting requires an explicitly configured language-specific model and is never claimed to be perfect.
- `MODEL_INFERRED` is not confirmed source content. High-risk medical/legal names, diagnoses, drugs, dosages, dates, provisions, reference numbers, and amounts are not contextually invented.
- Signatures, seals, stamps, and unreadable handwriting stay visible as source imagery and are never replaced as ordinary text.
- Folded, curved, severely warped, occluded, crossed-out, or physically missing content can remain unreadable and may require manual review.
- Machine translation is not certified medical, legal, or human translation.
- Text is replaced only where recognition, translation, and layout checks pass. Unsafe regions retain their original appearance; diagnostics remain separate from the primary document.
- Processing is local by default; no paid/cloud API is required and logs omit full document content.
            """
        )

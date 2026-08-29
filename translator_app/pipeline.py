"""Two-phase end-to-end orchestration independent of Streamlit widgets."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from .config.settings import Settings, get_settings
from .core.source_reconstruction import SourceReconstructor
from .core.document_loader import DocumentLoader
from .core.image_processor import ImagePreprocessor, PreprocessingOptions
from .core.ocr_normalization import OCRNormalizer
from .core.output_validation import validate_output
from .core.reconstruction_engine import DocumentReconstructionEngine
from .core.renderer import (
    render_debug_preview,
    render_enhanced_preview,
    render_output_preview,
    render_source_preview,
)
from .core.terminology import TerminologyProtector
from .core.translation_engine import TranslationService
from .core.block_grouping import group_validated_lines
from .core.script_detection import script_ratio
from .core.source_validation import (
    PageLanguageContext,
    SUPPORTED_TRANSLATION_LANGUAGES,
    build_page_context,
    is_translatable_block,
    normalize_language,
    readable_character_ratio,
    resolve_language,
    validate_source_block,
)
from .logging_config import configure_logging
from .models.model_manager import ModelManager
from .output_handler import make_audit_json, make_output_filename
from .exceptions import NoTranslationProducedError
from .schemas import (
    AnalysisResult,
    ProcessingOptions,
    ProcessingResult,
    ProcessingStage,
    ProcessingSummary,
    ProcessingStatus,
    ReconstructionMode,
    ReconstructionStatus,
    ReconstructionType,
    TranslationStatus,
    UncertaintyState,
    ScriptType,
)
from .utils.text_utils import normalize_text
from .utils.validation import validate_upload

LOGGER = logging.getLogger(__name__)
StageCallback = Callable[[ProcessingStage, float, str], None]


def _detect_domain(text_samples: list[str]) -> str:
    text = " ".join(text_samples[:100]).casefold()
    if re.search(r"\b(?:hospital|patient|diagnosis|prescription|dose|tablet|opd|ipd|mg|ml)\b", text):
        return "medical"
    if re.search(r"\b(?:court|case|petition|affidavit|section|fir|plaintiff|defendant)\b", text):
        return "legal"
    if re.search(r"\b(?:government|department|dispatch|office order|memorandum)\b", text):
        return "government"
    return "general"


class DocumentTranslationPipeline:
    """Orchestrate reviewable extraction first, then translation and reconstruction."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings)
        self.models = model_manager or ModelManager(self.settings)
        self.preprocessor = ImagePreprocessor(
            PreprocessingOptions(),
            default_upscale_factor=self.settings.ocr_upscale_factor,
            settings=self.settings,
        )
        self.loader = DocumentLoader(
            self.settings,
            self.models.get_page_ocr_pipeline,
            self.preprocessor,
        )
        self.reconstruction_engine = DocumentReconstructionEngine(self.settings)
        self.normalizer = OCRNormalizer()

    @staticmethod
    def _emit(
        callback: StageCallback | None,
        stage: ProcessingStage,
        progress: float,
        detail: str = "",
    ) -> None:
        if callback:
            callback(stage, max(0.0, min(1.0, progress)), detail or stage.value)

    def _default_options(self) -> ProcessingOptions:
        return ProcessingOptions(
            target_language=self.settings.default_target_language,
            ocr_languages=list(self.settings.ocr_languages or []),
            ocr_low_confidence_threshold=self.settings.ocr_low_confidence_threshold,
            handwriting_confidence_threshold=self.settings.handwriting_confidence_threshold,
            reconstruction_accept_threshold=self.settings.reconstruction_accept_threshold,
            auto_reconstruct_threshold=self.settings.auto_reconstruct_threshold,
            review_reconstruct_threshold=self.settings.review_reconstruct_threshold,
            min_context_quality=self.settings.min_context_quality,
            preprocessing_profile=self.settings.preprocessing_profile,
            ocr_upscale_factor=self.settings.ocr_upscale_factor,
            preserve_unreadable_handwriting_as_image=(
                self.settings.preserve_unreadable_handwriting_as_image
            ),
            handwriting_language_hint=self.settings.handwriting_language_hint,
            expected_source_language=self.settings.expected_source_language,
            reconstruction_mode=ReconstructionMode(self.settings.reconstruction_mode),
            document_domain=self.settings.document_domain,
        )

    def analyze(
        self,
        filename: str,
        data: bytes,
        options: ProcessingOptions | None = None,
        stage_callback: StageCallback | None = None,
    ) -> AnalysisResult:
        """Run through OCR and reconstruction, returning a state safe for human review."""
        started = time.perf_counter()
        options = options or self._default_options()
        if not options.ocr_languages:
            options.ocr_languages = list(self.settings.ocr_languages or ["eng"])
        self._emit(stage_callback, ProcessingStage.VALIDATING, 0.03)
        validated = validate_upload(filename, data, self.settings.max_upload_size)
        LOGGER.info(
            "Analyzing document id=%s format=%s bytes=%d",
            validated.sha256[:12],
            validated.file_format.value,
            len(data),
        )
        self._emit(stage_callback, ProcessingStage.READING, 0.08)

        def loader_progress(stage_name: str, fraction: float) -> None:
            if stage_name == "ocr":
                self._emit(stage_callback, ProcessingStage.OCR, 0.18 + fraction * 0.22)
            else:
                self._emit(stage_callback, ProcessingStage.EXTRACTING, 0.1 + fraction * 0.12)

        document = self.loader.load(validated, options, loader_progress)
        for block in document.blocks:
            self.normalizer.apply(block)
        self._emit(stage_callback, ProcessingStage.LANGUAGE, 0.43)
        language_detector = self.models.get_language_detector()
        language_detector.annotate_document(document)
        page_contexts: dict[int, PageLanguageContext] = {}
        for page in document.pages:
            quality_metrics = page.metadata.get("quality_metrics", {})
            ocr_context = build_page_context(
                page.blocks,
                handwriting_likelihood=page.handwriting_probability,
            )
            page.ocr_page_script = ocr_context.ocr_script
            page.ocr_page_script_confidence = ocr_context.ocr_confidence
            resolved_script = page.resolved_page_script
            resolved_confidence = page.resolved_page_script_confidence
            resolution_reason = page.script_resolution_reason
            if resolved_script == ScriptType.UNKNOWN:
                resolved_script = ocr_context.ocr_script
                resolved_confidence = ocr_context.ocr_confidence
                resolution_reason = (
                    "meaningful recognized Unicode evidence"
                    if resolved_script != ScriptType.UNKNOWN
                    else "visual and meaningful Unicode evidence were inconclusive"
                )
                page.resolved_page_script = resolved_script
                page.resolved_page_script_confidence = resolved_confidence
                page.script_resolution_reason = resolution_reason
            context = build_page_context(
                page.blocks,
                handwriting_likelihood=page.handwriting_probability,
                visual_script=page.visual_page_script,
                visual_confidence=page.visual_page_script_confidence,
                resolved_script=resolved_script,
                resolved_confidence=resolved_confidence,
                resolution_reason=resolution_reason,
                expected_language_prior=page.expected_language_prior,
            )
            page_contexts[page.page_number] = context
            page.metadata["dominant_script"] = context.dominant_script.value
            page.metadata["dominant_script_confidence"] = context.confidence
            page.metadata["ocr_page_script"] = ocr_context.ocr_script.value
            page.metadata["ocr_page_script_confidence"] = ocr_context.ocr_confidence
            page.metadata["resolved_page_script"] = context.dominant_script.value
            page.metadata["script_resolution_reason"] = context.resolution_reason
            LOGGER.info(
                "Page routing page=%d expected=%s visual_script=%s visual_confidence=%.3f "
                "ocr_script=%s ocr_confidence=%.3f resolved_script=%s handwriting_probability=%.3f "
                "page_type=%s line_regions=%d pa_htr_routes=%d hi_htr_routes=%d "
                "printed_routes=%d rejected_noise=%d",
                page.page_number,
                page.expected_language_prior,
                page.visual_page_script.value,
                page.visual_page_script_confidence,
                page.ocr_page_script.value,
                page.ocr_page_script_confidence,
                page.resolved_page_script.value,
                page.handwriting_probability,
                page.page_visual_type.value,
                int(page.metadata.get("detected_text_line_count", 0)),
                int(page.metadata.get("punjabi_htr_routes", 0)),
                int(page.metadata.get("hindi_htr_routes", 0)),
                int(page.metadata.get("printed_ocr_routes", 0)),
                int(page.metadata.get("rejected_noise_regions", 0)),
            )
            htr_unavailable_counts: dict[tuple[str, str], int] = {}
            for block in page.blocks:
                validation = validate_source_block(block, self.settings, context)
                unreviewed_htr_unavailable = bool(
                    block.metadata.get("htr_unavailable")
                    and not block.metadata.get("human_reviewed")
                )
                log_source_route = (
                    LOGGER.debug if unreviewed_htr_unavailable else LOGGER.info
                )
                log_source_route(
                    "Source routing page=%d block=%s region=%s visual_script=%s "
                    "recognized_script=%s resolved_script=%s language=%s linguistic_evidence=%.3f "
                    "ocr_confidence=%s text_quality=%.3f valid=%s reason=%s",
                    block.page_number,
                    block.block_id[:10],
                    block.region_type.value,
                    block.region_visual_script.value,
                    block.recognized_unicode_script.value,
                    block.resolved_script.value,
                    block.detected_language,
                    block.linguistic_evidence_score,
                    f"{block.ocr_confidence:.3f}" if block.ocr_confidence is not None else "native",
                    validation.quality,
                    validation.valid,
                    validation.reason,
                )
                if unreviewed_htr_unavailable:
                    key = (normalize_language(block.detected_language), validation.reason)
                    htr_unavailable_counts[key] = htr_unavailable_counts.get(key, 0) + 1
            for (language, reason), count in sorted(htr_unavailable_counts.items()):
                LOGGER.info(
                    "Source routing summary page=%d issue=HTR_UNAVAILABLE language=%s "
                    "count=%d action=manual_source_review reason=%s",
                    page.page_number,
                    language,
                    count,
                    reason,
                )
        domain = (
            _detect_domain([block.normalized_text for block in document.blocks])
            if options.document_domain == "auto"
            else options.document_domain
        )
        options.document_domain = domain
        document.metadata["document_domain"] = domain
        document.metadata["reconstruction_mode"] = options.reconstruction_mode.value
        document.metadata["routing_debug"] = options.routing_debug
        document.metadata["primary_output_policy"] = "source_template_only"
        document.metadata["diagnostics_embedded_in_primary"] = False
        self._emit(stage_callback, ProcessingStage.RECONSTRUCTION, 0.52)
        if options.enable_reconstruction:
            reconstructor = SourceReconstructor(
                self.settings,
                self.models.get_source_reconstruction_provider(),
                auto_threshold=options.auto_reconstruct_threshold,
                review_threshold=options.review_reconstruct_threshold,
                minimum_context_quality=options.min_context_quality,
                document_domain=domain,
            )
            for page in document.pages:
                eligible = [
                    block
                    for block in page.blocks
                    if block.is_ocr
                    and normalize_language(block.detected_language)
                    in SUPPORTED_TRANSLATION_LANGUAGES
                    and block.linguistic_evidence_score >= 0.20
                    and block.region_type.value not in {
                        "signature",
                        "stamp_seal",
                        "graphical_content",
                    }
                    and not block.metadata.get("handwriting_unsupported")
                ]
                reconstructor.process_blocks(
                    eligible, page_context=page_contexts[page.page_number]
                )
                for block in eligible:
                    if block.uncertainty_state == UncertaintyState.RECONSTRUCTED:
                        validate_source_block(block, self.settings, page_contexts[page.page_number])
        else:
            for block in document.blocks:
                if (
                    block.ocr_confidence is not None
                    and block.ocr_confidence < options.ocr_low_confidence_threshold
                ):
                    block.uncertainty_state = UncertaintyState.FLAGGED
                    if not block.source_text.strip():
                        block.reconstruction_type = ReconstructionType.UNREADABLE
        unreliable_pages = 0
        for page in document.pages:
            prose = [
                block
                for block in page.blocks
                if block.is_ocr
                and block.region_type.value not in {"signature", "stamp_seal", "graphical_content"}
                and not block.validation_reason.startswith("structured token")
            ]
            unreliable = [block for block in prose if not block.source_validated]
            ratio = len(unreliable) / len(prose) if prose else 0.0
            page.metadata["unreliable_prose_ratio"] = ratio
            catastrophic = len(prose) >= 3 and ratio > self.settings.catastrophic_unreadable_ratio
            page.metadata["ocr_quality_too_low"] = catastrophic
            if catastrophic:
                unreliable_pages += 1
                for block in prose:
                    if block.detected_language != "en":
                        block.metadata["automatic_translation_aborted"] = True
                        block.translation_status = TranslationStatus.SKIPPED
                        if ProcessingStatus.TRANSLATION_SKIPPED not in block.processing_statuses:
                            block.processing_statuses.append(ProcessingStatus.TRANSLATION_SKIPPED)
                document.warnings.append(
                    f"Page {page.page_number}: OCR quality is too low for reliable automatic translation "
                    f"({len(unreliable)}/{len(prose)} prose regions failed source validation). "
                    "The original regions were preserved for review."
                )
            page.blocks = group_validated_lines(
                page.blocks,
                maximum_line_gap_ratio=self.settings.paragraph_line_gap_ratio,
            )
        document.metadata["ocr_quality_abort_page_count"] = unreliable_pages
        source_previews = render_source_preview(
            document, max_pages=self.settings.max_preview_pages
        )
        enhanced_previews = render_enhanced_preview(
            document, max_pages=self.settings.max_preview_pages
        )
        debug_previews = (
            render_debug_preview(document, max_pages=self.settings.max_preview_pages)
            if options.debug_bounding_boxes
            else []
        )
        if options.review_before_render:
            self._emit(stage_callback, ProcessingStage.REVIEW, 0.58)
        return AnalysisResult(
            document=document,
            options=options,
            started_at=started,
            source_preview_images=source_previews,
            enhanced_preview_images=enhanced_previews,
            debug_preview_images=debug_previews,
        )

    def apply_review_edits(
        self, analysis: AnalysisResult, reviewed_text: dict[str, str]
    ) -> None:
        """Validate explicitly confirmed source readings without altering raw evidence."""
        for block in analysis.document.blocks:
            if block.block_id not in reviewed_text:
                continue
            value = normalize_text(reviewed_text.get(block.block_id, ""))
            if not value:
                continue
            previous = block.effective_source_text
            if block.reconstruction_status == ReconstructionStatus.CANDIDATE_REVIEW:
                source = block.normalized_text or block.source_text
                start = block.metadata.get("missing_span_start")
                end = block.metadata.get("missing_span_end")
                if isinstance(start, int) and isinstance(end, int):
                    prefix = source[:start].rstrip()
                    suffix = source[end:].lstrip()
                    if (prefix and not value.startswith(prefix)) or (
                        suffix and not value.endswith(suffix)
                    ):
                        analysis.document.warnings.append(
                            f"Block {block.block_id[:10]}: manual missing-span confirmation was ignored "
                            "because text outside the bounded span changed."
                        )
                        continue
                    candidate_end = len(value) - len(suffix) if suffix else len(value)
                    block.reconstruction_candidate = value[len(prefix) : candidate_end].strip()
                    expected_script = (
                        ScriptType.GURMUKHI
                        if block.detected_language == "pa"
                        else ScriptType.DEVANAGARI
                        if block.detected_language == "hi"
                        else ScriptType.UNKNOWN
                    )
                    if (
                        expected_script == ScriptType.UNKNOWN
                        or script_ratio(block.reconstruction_candidate, expected_script) < 0.80
                    ):
                        analysis.document.warnings.append(
                            f"Block {block.block_id[:10]}: manual missing-span confirmation was ignored "
                            "because the candidate was not in the expected source script."
                        )
                        continue
            block.reconstructed_text = value
            block.reconstruction_confidence = 1.0
            block.uncertainty_state = UncertaintyState.RECONSTRUCTED
            block.reconstruction_type = ReconstructionType.MANUALLY_CONFIRMED
            block.reconstruction_status = ReconstructionStatus.MANUALLY_CONFIRMED
            action = "confirmed" if value == previous else "corrected"
            block.provenance.append(
                f"MANUALLY_CORRECTED: source-language reading explicitly {action} in UI; raw OCR retained"
            )
            block.metadata["human_reviewed"] = True
            block.metadata["manual_review_action"] = action
            block.metadata["source_origin"] = "manual_correction"
            block.metadata["correction_origin"] = "manual_correction"
            block.metadata.pop("handwriting_unsupported", None)
            block.metadata.pop("preserve_region_as_image", None)
            block.metadata.pop("automatic_translation_aborted", None)
            block.translation_status = TranslationStatus.PENDING
            block.source_validated = False
            resolution = resolve_language(
                value,
                block.script,
                PageLanguageContext(block.script, 1.0),
                1.0,
            )
            block.detected_language = resolution.language
            block.language_confidence = resolution.confidence
            block.script = resolution.script
            block.processing_statuses = [
                status
                for status in block.processing_statuses
                if status
                not in {
                    ProcessingStatus.UNREADABLE,
                    ProcessingStatus.TRANSLATION_SKIPPED,
                    ProcessingStatus.LANGUAGE_UNCERTAIN,
                    ProcessingStatus.HANDWRITING_UNSUPPORTED,
                    ProcessingStatus.HTR_UNAVAILABLE,
                    ProcessingStatus.RECONSTRUCTION_REVIEW_REQUIRED,
                }
            ]
            if ProcessingStatus.MANUALLY_CORRECTED not in block.processing_statuses:
                block.processing_statuses.append(ProcessingStatus.MANUALLY_CORRECTED)
            page = next(
                page
                for page in analysis.document.pages
                if page.page_number == block.page_number
            )
            try:
                dominant_script = page.resolved_page_script
                if dominant_script == ScriptType.UNKNOWN:
                    dominant_script = ScriptType(
                        page.metadata.get("dominant_script", block.script.value)
                    )
            except ValueError:
                dominant_script = block.script
            context = PageLanguageContext(
                dominant_script=dominant_script,
                confidence=float(page.metadata.get("dominant_script_confidence", 1.0)),
                handwriting_likelihood=float(
                    page.metadata.get("quality_metrics", {}).get("handwriting_likelihood", 0.0)
                ),
            )
            validate_source_block(block, self.settings, context)
            if block.source_validated:
                block.uncertainty_state = UncertaintyState.CONFIRMED
                block.validation_reason = f"explicit human-{action} source reading; safety gates passed"

    def _translate(
        self,
        analysis: AnalysisResult,
        stage_callback: StageCallback | None,
    ) -> None:
        document, options = analysis.document, analysis.options
        self._emit(stage_callback, ProcessingStage.TRANSLATION, 0.64)
        eligible: list = []
        skipped_by_reason: dict[str, int] = {}
        skipped_unidentified = 0
        for block in document.blocks:
            language = normalize_language(block.detected_language)
            block.detected_language = language
            if language == "en" and block.source_validated:
                block.english_translation = block.effective_source_text
                block.translation_status = TranslationStatus.NOT_REQUIRED
            elif is_translatable_block(block) and not block.metadata.get("automatic_translation_aborted"):
                eligible.append(block)
            else:
                block.translation_status = TranslationStatus.SKIPPED
                reason = block.validation_reason or "source block did not pass translation gates"
                skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                if language == "und":
                    skipped_unidentified += 1
                if ProcessingStatus.TRANSLATION_SKIPPED not in block.processing_statuses:
                    block.processing_statuses.append(ProcessingStatus.TRANSLATION_SKIPPED)
        if eligible:
            try:
                provider = self.models.get_translation_provider()
                protector = TerminologyProtector(options.protected_terms, options.glossary)
                document.warnings.extend(
                    TranslationService(provider, self.settings, protector).translate_blocks(
                        eligible, options.target_language
                    )
                )
            except Exception:
                LOGGER.warning("Translation provider unavailable", exc_info=True)
                document.warnings.append(
                    "The local translation model was unavailable; untranslated source text was retained."
                )
                for block in eligible:
                    block.translation_status = TranslationStatus.FAILED
        for block in eligible:
            if block.translation_status == TranslationStatus.TRANSLATED:
                if ProcessingStatus.TRANSLATED not in block.processing_statuses:
                    block.processing_statuses.append(ProcessingStatus.TRANSLATED)
        if skipped_by_reason:
            if skipped_unidentified:
                document.warnings.append(
                    f"{skipped_unidentified} region(s) could not be reliably language-identified and were preserved for review."
                )
            other = sum(skipped_by_reason.values()) - skipped_unidentified
            if other:
                document.warnings.append(
                    f"{other} additional region(s) failed source-quality or safety gates and were not translated."
                )

    def finalize(
        self,
        analysis: AnalysisResult,
        stage_callback: StageCallback | None = None,
        *,
        require_translation: bool = False,
    ) -> ProcessingResult:
        """Translate reviewed state, rebuild the document, validate, and create exports."""
        document, options = analysis.document, analysis.options
        self._translate(analysis, stage_callback)
        translated_count = sum(
            block.translation_status == TranslationStatus.TRANSLATED
            for block in document.blocks
        )
        if require_translation and translated_count == 0:
            validated_indic_count = sum(
                block.source_validated
                and normalize_language(block.detected_language)
                in SUPPORTED_TRANSLATION_LANGUAGES
                for block in document.blocks
            )
            if validated_indic_count:
                user_message = (
                    "Validated Punjabi/Hindi source text was available, but the local translation "
                    "model produced no English output. Check the model status and logs, then retry; "
                    "no unchanged file was exported as a translated document."
                )
                technical_reason = (
                    "Translation-required finalization produced zero translated blocks despite "
                    f"{validated_indic_count} validated Indic source block(s)"
                )
            else:
                user_message = NoTranslationProducedError.user_message
                technical_reason = (
                    "Translation-required finalization had no validated Punjabi/Hindi source blocks"
                )
            raise NoTranslationProducedError(
                technical_reason,
                user_message=user_message,
            )
        self._emit(stage_callback, ProcessingStage.LAYOUT, 0.76)
        self._emit(stage_callback, ProcessingStage.RENDERING, 0.83)
        # ``TRANSLATED`` proves only that a translation provider returned text.
        # It does not prove that the renderer could safely alter the primary
        # document.  Clear any marker from a previous reconstruction attempt so
        # the selected backend must explicitly commit each applied replacement.
        for block in document.blocks:
            block.metadata["replacement_applied"] = False
        output_bytes, extension, output_mime = self.reconstruction_engine.rebuild(document)
        applied_replacement_count = sum(
            block.translation_status == TranslationStatus.TRANSLATED
            and bool((block.english_translation or "").strip())
            and block.metadata.get("replacement_applied") is True
            for block in document.blocks
        )
        document.metadata["applied_translation_replacement_count"] = (
            applied_replacement_count
        )
        if require_translation and applied_replacement_count == 0:
            reasons: dict[str, int] = {}
            for block in document.blocks:
                if block.translation_status != TranslationStatus.TRANSLATED:
                    continue
                reason = str(
                    block.metadata.get(
                        "primary_output_replacement_reason",
                        "renderer_did_not_commit_replacement",
                    )
                )
                reasons[reason] = reasons.get(reason, 0) + 1
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reasons.items())
            ) or "no renderer commit marker"
            raise NoTranslationProducedError(
                "Translation-required reconstruction applied zero replacements "
                f"after {translated_count} translated block(s): {reason_summary}",
                user_message=(
                    "English translation text was produced, but none of it could be safely "
                    "placed into the source layout. The unchanged source was not exported as "
                    "a translated document. Review the layout/replacement reasons and retry."
                ),
            )
        output_filename = make_output_filename(document.source_filename, extension)
        self._emit(stage_callback, ProcessingStage.VALIDATING_OUTPUT, 0.93)
        document.warnings.extend(validate_output(output_bytes, extension, document))
        document.warnings = list(dict.fromkeys(document.warnings))
        output_previews = render_output_preview(
            output_bytes, extension, max_pages=self.settings.max_preview_pages
        )
        elapsed = time.perf_counter() - analysis.started_at
        blocks = document.blocks
        languages = sorted(
            {
                block.detected_language
                for block in blocks
                if block.detected_language not in {"und", ""}
            }
        )
        summary = ProcessingSummary(
            filename=document.source_filename,
            file_type=document.file_format.value,
            page_count=len(document.pages),
            detected_languages=languages,
            text_block_count=len(blocks),
            ocr_block_count=sum(block.is_ocr for block in blocks),
            low_confidence_ocr_count=sum(
                block.is_ocr
                and block.ocr_confidence is not None
                and block.ocr_confidence < options.ocr_low_confidence_threshold
                for block in blocks
            ),
            reconstructed_block_count=sum(
                block.uncertainty_state == UncertaintyState.RECONSTRUCTED
                for block in blocks
            ),
            uncertain_block_count=sum(block.is_uncertain for block in blocks),
            translation_count=sum(
                block.translation_status == TranslationStatus.TRANSLATED for block in blocks
            ),
            layout_overflow_count=sum(
                block.layout_status.value == "overflow" for block in blocks
            ),
            processing_duration_seconds=elapsed,
            output_filename=output_filename,
            region_count=sum(len(page.regions) for page in document.pages),
            handwriting_block_count=sum(block.is_handwritten for block in blocks),
            unreadable_block_count=sum(
                block.reconstruction_type == ReconstructionType.UNREADABLE for block in blocks
            ),
            preprocessing_profiles=sorted(
                {
                    str(page.metadata.get("preprocessing_profile"))
                    for page in document.pages
                    if page.metadata.get("preprocessing_profile")
                }
            ),
            warnings=document.warnings,
            dominant_scripts=sorted(
                {
                    str(page.metadata.get("dominant_script"))
                    for page in document.pages
                    if page.metadata.get("dominant_script") not in {None, "", "unknown"}
                }
            ),
            printed_region_count=sum(
                region.region_type.value == "printed_text"
                for page in document.pages
                for region in page.regions
            ),
            handwritten_region_count=sum(
                region.region_type.value == "handwriting"
                for page in document.pages
                for region in page.regions
            ),
            validated_punjabi_line_count=sum(
                block.source_validated and block.detected_language == "pa" for block in blocks
            ),
            validated_hindi_line_count=sum(
                block.source_validated and block.detected_language == "hi" for block in blocks
            ),
            translation_skipped_count=sum(
                block.translation_status == TranslationStatus.SKIPPED for block in blocks
            ),
            htr_recognized_line_count=sum(
                ProcessingStatus.HTR_RECOGNIZED in block.processing_statuses for block in blocks
            ),
            manually_reviewed_line_count=sum(
                ProcessingStatus.MANUALLY_CORRECTED in block.processing_statuses for block in blocks
            ),
            rejected_handwriting_line_count=sum(
                block.is_handwritten and not block.source_validated for block in blocks
            ),
            missing_span_count=sum(block.missing_span_detected for block in blocks),
            auto_reconstructed_span_count=sum(
                block.reconstruction_status == ReconstructionStatus.AUTO_ACCEPTED
                for block in blocks
            ),
            review_reconstruction_count=sum(
                block.reconstruction_status == ReconstructionStatus.CANDIDATE_REVIEW
                for block in blocks
            ),
            manually_confirmed_count=sum(
                block.reconstruction_status == ReconstructionStatus.MANUALLY_CONFIRMED
                for block in blocks
            ),
            unresolved_missing_span_count=sum(
                block.missing_span_detected
                and block.reconstruction_status
                in {
                    ReconstructionStatus.BLOCKED,
                    ReconstructionStatus.REJECTED,
                    ReconstructionStatus.UNAVAILABLE,
                }
                for block in blocks
            ),
            punjabi_htr_route_count=sum(
                int(page.metadata.get("punjabi_htr_routes", 0))
                for page in document.pages
            ),
            hindi_htr_route_count=sum(
                int(page.metadata.get("hindi_htr_routes", 0))
                for page in document.pages
            ),
            printed_ocr_route_count=sum(
                int(page.metadata.get("printed_ocr_routes", 0))
                for page in document.pages
            ),
            rejected_noise_region_count=sum(
                int(page.metadata.get("rejected_noise_regions", 0))
                for page in document.pages
            ),
            page_visual_types=sorted(
                {
                    page.page_visual_type.value
                    for page in document.pages
                    if page.page_visual_type.value != "unknown"
                }
            ),
        )
        self._emit(stage_callback, ProcessingStage.COMPLETE, 1.0)
        LOGGER.info(
            "Completed document id=%s pages=%d blocks=%d seconds=%.2f",
            document.document_id[:12],
            len(document.pages),
            len(blocks),
            elapsed,
        )
        return ProcessingResult(
            document=document,
            output_bytes=output_bytes,
            output_filename=output_filename,
            output_mime_type=output_mime,
            summary=summary,
            source_preview_images=analysis.source_preview_images,
            enhanced_preview_images=analysis.enhanced_preview_images,
            debug_preview_images=analysis.debug_preview_images,
            output_preview_images=output_previews,
            audit_json=make_audit_json(document),
        )

    def process(
        self,
        filename: str,
        data: bytes,
        options: ProcessingOptions | None = None,
        stage_callback: StageCallback | None = None,
    ) -> ProcessingResult:
        analysis = self.analyze(filename, data, options, stage_callback)
        return self.finalize(analysis, stage_callback)

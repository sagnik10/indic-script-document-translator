"""Compare overlapping OCR candidates without discarding low-confidence evidence."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..schemas import ScriptType, TextBlock
from .script_detection import (
    dominant_script,
    linguistic_evidence_score,
    script_ratio,
)
from .source_validation import calculate_text_quality


def _text_quality(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    meaningful = sum(character.isalnum() or ord(character) > 127 for character in visible)
    replacement_penalty = text.count("�") + text.count("?") * 0.25
    return max(0.0, min(1.0, meaningful / len(visible) - replacement_penalty / len(visible)))


class OCRCandidateComparator:
    """Select by confidence, text sanity, and agreement across OCR configurations."""

    @staticmethod
    def _groups(candidates: list[TextBlock]) -> list[list[TextBlock]]:
        groups: list[list[TextBlock]] = []
        for candidate in sorted(candidates, key=lambda block: (block.source_bbox.y0, block.source_bbox.x0)):
            matching = next(
                (
                    group
                    for group in groups
                    if any(
                        candidate.source_bbox.intersection_ratio(existing.source_bbox) >= 0.35
                        for existing in group
                    )
                ),
                None,
            )
            if matching is None:
                groups.append([candidate])
            else:
                matching.append(candidate)
        return groups

    def choose(
        self,
        candidates: list[TextBlock],
        expected_script: ScriptType | None = None,
    ) -> list[TextBlock]:
        selected: list[TextBlock] = []
        for group in self._groups(candidates):
            group_text = " ".join(candidate.source_text for candidate in group)
            group_script, expected_ratio = dominant_script(group_text)
            expected_name = (
                expected_script.value
                if expected_script is not None
                else group_script
            )
            scored: list[tuple[float, TextBlock]] = []
            for candidate in group:
                similarities = [
                    SequenceMatcher(
                        None,
                        candidate.source_text.casefold(),
                        other.source_text.casefold(),
                    ).ratio()
                    for other in group
                    if other is not candidate
                ]
                consensus = max(similarities, default=0.5)
                confidence = candidate.ocr_confidence or 0.0
                script_consistency = (
                    script_ratio(candidate.source_text, expected_name)
                    if expected_name != "unknown"
                    and (expected_script is not None or expected_ratio >= 0.25)
                    else 0.5
                )
                quality = calculate_text_quality(candidate.source_text, expected_name)
                linguistic_evidence = linguistic_evidence_score(
                    candidate.source_text, expected_name
                )
                score = (
                    confidence * 0.34
                    + consensus * 0.14
                    + quality * 0.16
                    + script_consistency * 0.20
                    + linguistic_evidence * 0.16
                )
                if expected_script in {ScriptType.GURMUKHI, ScriptType.DEVANAGARI}:
                    score *= 0.15 + 0.85 * script_consistency
                scored.append((score, candidate))
            _score, winner = max(scored, key=lambda item: item[0])
            winner.ocr_alternatives = [
                {
                    "text": candidate.source_text,
                    "confidence": candidate.ocr_confidence,
                    "engine": candidate.ocr_engine,
                    "bbox": {
                        "x0": candidate.source_bbox.x0,
                        "y0": candidate.source_bbox.y0,
                        "x1": candidate.source_bbox.x1,
                        "y1": candidate.source_bbox.y1,
                    },
                }
                for _candidate_score, candidate in sorted(scored, key=lambda item: item[0], reverse=True)
                if candidate is not winner
            ]
            winner.metadata["ensemble_candidate_count"] = len(group)
            winner.metadata["ensemble_score"] = round(_score, 4)
            winner.metadata["ensemble_expected_script"] = expected_name
            winner.metadata["ensemble_script_consistency"] = round(
                script_ratio(winner.source_text, expected_name)
                if expected_name != "unknown"
                else 0.0,
                4,
            )
            winner.provenance.append(
                f"OCR ensemble selected {winner.ocr_engine} from {len(group)} candidate(s)"
            )
            selected.append(winner)
        return sorted(selected, key=lambda block: (block.source_bbox.y0, block.source_bbox.x0))

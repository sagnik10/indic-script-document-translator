"""Structured-value and user-glossary protection before machine translation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.text_utils import STRUCTURED_TOKEN_PATTERN, restore_structured_tokens


DOMAIN_TOKEN_PATTERN = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|kg|ml|mL|IU|units?|tablets?|capsules?)\b|"
    r"\b(?:Sec(?:tion)?\.?|FIR|Case|File|Ref(?:erence)?|Dispatch|MRD|OPD|IPD)\s*"
    r"(?:No\.?\s*)?[A-Za-z0-9_./()-]+|"
    r"\b(?:[A-Z][A-Za-z.&'-]+\s+){0,5}(?:Hospital|Clinic|Medical College|Court|Department)\b|"
    r"\b[A-Z]{2,8}(?:[-/]?[A-Z0-9]{1,12})*\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ProtectedText:
    text: str
    replacements: dict[str, str]
    source_tokens: tuple[str, ...]


class TerminologyProtector:
    """Protect exact structured values and restore glossary-approved targets."""

    def __init__(
        self,
        protected_terms: list[str] | None = None,
        glossary: dict[str, str] | None = None,
    ) -> None:
        self.protected_terms = sorted(
            {term.strip() for term in (protected_terms or []) if term.strip()},
            key=len,
            reverse=True,
        )
        self.glossary = {
            source.strip(): target.strip()
            for source, target in (glossary or {}).items()
            if source.strip() and target.strip()
        }

    @staticmethod
    def _placeholder(index: int) -> str:
        # Letter-only sentinels avoid being mistaken for dates, numbers, or identifiers.
        value = index
        letters = ""
        while True:
            letters = chr(ord("a") + value % 26) + letters
            value = value // 26 - 1
            if value < 0:
                break
        return f"__dtxprotected_{letters}__"

    def protect(self, text: str) -> ProtectedText:
        replacements: dict[str, str] = {}
        source_tokens: list[str] = []
        output = text
        explicit = self.protected_terms + list(self.glossary)
        for term in explicit:
            pattern = re.compile(re.escape(term), re.IGNORECASE)

            def replace_explicit(match: re.Match[str], source_term: str = term) -> str:
                placeholder = self._placeholder(len(replacements))
                replacements[placeholder] = self.glossary.get(source_term, match.group(0))
                source_tokens.append(match.group(0))
                return placeholder

            output = pattern.sub(replace_explicit, output)

        combined = re.compile(
            f"(?:{STRUCTURED_TOKEN_PATTERN.pattern})|(?:{DOMAIN_TOKEN_PATTERN.pattern})",
            re.IGNORECASE,
        )

        def replace_automatic(match: re.Match[str]) -> str:
            placeholder = self._placeholder(len(replacements))
            replacements[placeholder] = match.group(0)
            source_tokens.append(match.group(0))
            return placeholder

        output = combined.sub(replace_automatic, output)
        return ProtectedText(output, replacements, tuple(source_tokens))

    @staticmethod
    def restore(text: str, protected: ProtectedText) -> str:
        return restore_structured_tokens(text, protected.replacements)

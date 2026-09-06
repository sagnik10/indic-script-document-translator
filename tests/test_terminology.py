from translator_app.core.terminology import TerminologyProtector


def test_structured_medical_legal_tokens_and_glossary_are_restored_exactly() -> None:
    protector = TerminologyProtector(
        protected_terms=["Civil Hospital"],
        glossary={"ਤਹਿਸੀਲ": "Tehsil"},
    )
    source = "ਤਹਿਸੀਲ Civil Hospital Case No. PB/31/2016 dose 5 mg on 31.03.2016"
    protected = protector.protect(source)
    assert "31.03.2016" not in protected.text
    assert "5 mg" not in protected.text
    restored = protector.restore(f"English {protected.text}", protected)
    assert "Tehsil" in restored
    assert "Civil Hospital" in restored
    assert "PB/31/2016" in restored
    assert "5 mg" in restored
    assert "31.03.2016" in restored


def test_placeholders_are_not_recursively_protected() -> None:
    protector = TerminologyProtector(protected_terms=["PGIMER"])
    protected = protector.protect("PGIMER 04.09.2008")
    assert len(protected.replacements) == 2
    assert protector.restore(protected.text, protected) == "PGIMER 04.09.2008"


from translator_app.reconstruction.docx_reconstructor import (
    replace_paragraph_text_preserving_runs,
)


class FakeRun:
    def __init__(self, text: str, style: str) -> None:
        self.text = text
        self.style = style


class FakeParagraph:
    def __init__(self) -> None:
        self.runs = [FakeRun("Original ", "plain"), FakeRun("bold", "bold")]

    def add_run(self, text: str):
        run = FakeRun(text, "plain")
        self.runs.append(run)
        return run


def test_run_objects_and_styles_survive_text_replacement() -> None:
    paragraph = FakeParagraph()
    first, second = paragraph.runs
    replace_paragraph_text_preserving_runs(paragraph, "Translated styled content")
    assert paragraph.runs[0] is first
    assert paragraph.runs[1] is second
    assert first.style == "plain"
    assert second.style == "bold"
    assert " ".join("".join(run.text for run in paragraph.runs).split()) == (
        "Translated styled content"
    )

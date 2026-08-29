from translator_app.schemas import ProcessingStage
from translator_app.ui.components import create_progress_callback


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def progress(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))

    def update(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))

    def markdown(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def test_review_stage_is_presented_as_complete_not_still_processing() -> None:
    progress, status, stage_area = _Recorder(), _Recorder(), _Recorder()
    callback = create_progress_callback(progress, status, stage_area)

    callback(ProcessingStage.REVIEW, 0.57, "Reviewing uncertain OCR")

    assert progress.calls[-1] == (
        (1.0,),
        {"text": "OCR analysis complete — source review required"},
    )
    assert status.calls[-1][1] == {
        "label": "OCR analysis complete — review below",
        "state": "complete",
        "expanded": False,
    }

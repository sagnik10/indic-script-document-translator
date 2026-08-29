"""Regenerate the self-contained Kaggle notebook from ``train.py``."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    source = (HERE / "train.py").read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Gurmukhi handwriting recognizer (free GPU)\n",
                    "This notebook downloads only the public IIIT Gurmukhi word corpus. "
                    "Do not upload private documents. Select a free GPU accelerator, enable "
                    "Internet, and run all cells. The final ZIP is enabled only when held-out "
                    "source-script validation passes.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%pip install -q 'accelerate>=1.1,<2' 'transformers>=4.45,<5' "
                    "'safetensors>=0.4,<1' 'sentencepiece>=0.2,<1'\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    f"training_source = {source!r}\n",
                    "script_path = Path('/kaggle/working/gurmukhi_train.py')\n",
                    "script_path.write_text(training_source, encoding='utf-8')\n",
                    "print(f'Prepared {script_path}')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import subprocess, sys\n",
                    "from pathlib import Path\n",
                    "checkpoint_candidates = sorted(\n",
                    "    Path('/kaggle/input').rglob('checkpoint-*'),\n",
                    "    key=lambda p: int(p.name.rsplit('-', 1)[-1]) if p.name.rsplit('-', 1)[-1].isdigit() else -1,\n",
                    ")\n",
                    "command = [sys.executable, '/kaggle/working/gurmukhi_train.py']\n",
                    "if checkpoint_candidates:\n",
                    "    command += ['--resume-from', str(checkpoint_candidates[-1])]\n",
                    "print('Starting/resuming source-language HTR training')\n",
                    "subprocess.run(command, check=True)\n",
                ],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = HERE / "kaggle_train.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

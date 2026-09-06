# Automatic Gurmukhi HTR training

This folder trains the missing Punjabi handwriting recognizer without sending
private documents to any service. It downloads the public IIIT-INDIC-HW-WORDS
Gurumukhi training archive, combines authentic handwritten word crops into
line-level examples, fine-tunes `microsoft/trocr-small-stage1`, and exports a
Hugging Face-compatible source-transcription model.

The model is not allowed into the production route merely because training
finishes. The export must pass held-out character-error-rate, Gurmukhi Unicode
purity, and non-empty-output gates. Runtime confidence, script-consistency, and
source-quality gates remain active afterward.

## Free Kaggle run

1. Sign in to a free Kaggle account. No paid plan or billing details are needed.
2. Import `kaggle_train.ipynb` as a new **private** notebook.
3. In Notebook options, choose a free GPU accelerator and enable Internet.
4. Run all cells. Do not attach `Test1.jpeg` or any private document.
5. When the run finishes, download
   `/kaggle/working/gurmukhi_htr/gurmukhi_htr_model.zip` from notebook outputs.

Checkpoints are written every 500 steps. If a free session ends, save the
notebook version, attach its output to the next run, and rerun the notebook; it
automatically selects the newest attached `checkpoint-*` directory.

## Install the result

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe training\gurmukhi_htr\install_model.py "C:\path\to\gurmukhi_htr_model.zip"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The installer verifies the manifest, validation result, critical model files,
and checksums. A previous model is renamed to a timestamped backup. On restart,
the model manager discovers the validated bundle under
`.runtime/models/gurmukhi_htr` automatically.

## Limitations

The public corpus is word-level and does not represent every historical,
photocopied, legal, or medical handwriting style. Synthetic line composition
reduces the app/training shape mismatch but does not eliminate domain shift.
Consequently, automatic output must still pass the application's Gurmukhi
Unicode, confidence, and source-quality validation before translation. Failed
or unreadable lines remain preserved as source pixels.

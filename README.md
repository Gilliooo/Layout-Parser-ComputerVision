# Layout Parser + Handwriting Recognition

Detects layout regions (titles, paragraphs, lists) in handwritten note images using **DocLayout-YOLO**, then runs a custom character-level OCR model to convert them into structured Markdown text.

## Requirements

- Python 3.11 — TensorFlow does not support 3.12+ on Windows yet
- Git

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Gilliooo/Layout-Parser-ComputerVision.git
cd Layout-Parser-ComputerVision
```

### 2. Install Python 3.11

**Windows:**
```powershell
winget install Python.Python.3.11
```

**Mac / Linux:** Download from [python.org](https://www.python.org/downloads/release/python-3119/)

### 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
py -3.11 -m venv .venv --without-pip
.venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
```

**Mac / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the app

```bash
python -m streamlit run app.py
```

On first run with **DocLayout-YOLO** mode selected, the model weights (~100 MB) are automatically downloaded from Hugging Face and cached locally.

## Notes

- `handwriting_recognition_model.keras` and `classes.json` are included in the repo — no manual download needed.
- The DocLayout-YOLO weights are fetched from [`juliozhao/DocLayout-YOLO-DocStructBench`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) on first use.
- Use `python -m streamlit` instead of just `streamlit` to ensure the venv's Python is used.

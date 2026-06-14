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

## Deploy to Streamlit Community Cloud

This repo is already set up to deploy as a live website on [Streamlit Community Cloud](https://share.streamlit.io) (free).

1. Push this repo to GitHub (it already lives at `Gilliooo/Layout-Parser-ComputerVision`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** `Gilliooo/Layout-Parser-ComputerVision`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and set **Python version** to **3.12**. ⚠️ **This is required** — leaving it on the default (currently Python 3.14) breaks the build because TensorFlow has no 3.14 wheels yet. Use **3.12** (3.11 or 3.13 also work; do **not** use 3.14).
5. Click **Deploy**. The first build takes a few minutes while it installs the dependencies and the first DocLayout-YOLO run downloads the ~100 MB weights.

### Already deployed and stuck on Python 3.14?

If you created the app before pinning the version, the build will fail with
`Could not find a version that satisfies the requirement tensorflow-cpu`.
Fix it without recreating the app:

1. Open the app → **Manage app** (bottom-right) → **⚙️ Settings** → **General**.
2. Set **Python version** to **3.12** → **Save**.
3. **Reboot app** from the same menu.

(If the dropdown isn't available for an existing app, delete the app and redeploy, setting Python 3.12 in **Advanced settings** during creation.)

### What makes it cloud-ready

- **`requirements.txt`** pins CPU-only `torch`/`torchvision` (via the PyTorch CPU index) and `tensorflow-cpu` so the build fits within Streamlit Cloud's disk/RAM limits.
- **`packages.txt`** installs `libgl1` + `libglib2.0-0`, the system libraries OpenCV needs on the headless Linux runners.
- **`.streamlit/config.toml`** sets the theme and a 20 MB upload limit.
- The app shows detailed **loading screens** (boot overlay → model warm-up → per-region progress) so users always see what's happening during the slow steps.

> **Note on resources:** the free tier runs TensorFlow + PyTorch together, which is memory-heavy. If the app gets killed for exceeding memory, use **No layout** mode (skips the YOLO model) or upgrade the Streamlit Cloud resources.

## Notes

- `handwriting_recognition_model.keras` and `classes.json` are included in the repo — no manual download needed.
- The DocLayout-YOLO weights are fetched from [`juliozhao/DocLayout-YOLO-DocStructBench`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) on first use.
- Use `python -m streamlit` instead of just `streamlit` to ensure the venv's Python is used.

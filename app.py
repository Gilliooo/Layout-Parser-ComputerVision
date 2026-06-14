import time

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from doclayout_module import (load_doclayout_model, detect_layout_doclayout, crop_rgb as crop_rgb_doc,
)

from ocr_module import (load_handwriting_model,load_classes, ocr_char_level,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Notes to Structured Text",
    layout="wide",
)

# A little CSS to make it feel like a real website rather than a raw script.
st.markdown(
    """
    <style>
      /* hide the anchor "link" icon / clickable button next to headings */
      [data-testid="stHeaderActionElements"] { display: none !important; }
      h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a,
      .stMarkdown a.anchor-link { display: none !important; }

      /* hero header */
      .hero {
          background: #000000;
          border: 1px solid #22c55e;
          padding: 2.2rem 2.4rem;
          border-radius: 18px;
          color: #e6f4ea;
          margin-bottom: 1.4rem;
          box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
      }
      .hero h1 { color:#f4fff8; margin:0 0 .4rem 0; font-size: 2.1rem; line-height:1.15; }
      .hero p  { color: rgba(230,244,234,0.85); margin:0; font-size: 1.02rem; }
      .hero h1 b, .hero p b { color:#22c55e; }

      .badge {
          display:inline-block; background: rgba(34,197,94,0.15);
          border: 1px solid rgba(34,197,94,0.35); color:#7ee2a8;
          padding: .25rem .7rem; border-radius: 999px; font-size:.8rem;
          margin-right:.4rem; margin-top:.7rem;
      }

      /* full-screen loading overlay */
      #boot-overlay {
          position: fixed; inset: 0; z-index: 9999;
          background: linear-gradient(135deg, #08130d 0%, #0e2a1c 100%);
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          color:#e6f4ea; font-family: sans-serif; transition: opacity .4s ease;
      }
      #boot-overlay .spinner {
          width: 54px; height: 54px; border: 5px solid rgba(34,197,94,.25);
          border-top-color:#22c55e; border-radius:50%; animation: spin 0.9s linear infinite;
          margin-bottom: 1.1rem;
      }
      #boot-overlay h2 { margin:.2rem 0; font-weight:700; color:#f4fff8; }
      #boot-overlay p  { margin:.15rem 0; opacity:.85; font-size:.92rem; }
      @keyframes spin { to { transform: rotate(360deg); } }

      .footer { color:#5b6b60; font-size:.82rem; text-align:center; margin-top:2.5rem; }
      .stCodeBlock { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# A boot overlay that shows while Streamlit downloads the page / spins up the
# server. It self-removes once the script has rendered (so the user always sees
# something with detail instead of a blank screen on a cold start).
boot_overlay = st.empty()
boot_overlay.markdown(
    """
    <div id="boot-overlay">
      <div class="spinner"></div>
      <h2>Starting the app...</h2>
      <p>Booting the Streamlit server and importing the vision libraries.</p>
      <p>The first cold start can take ~30-60s, hang tight.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
    <div class="hero">
      <h1>Handwritten Notes to Structured Text</h1>
      <p>Upload a photo of handwritten notes. We detect the layout (titles, paragraphs, lists)
         with <b>DocLayout-YOLO</b>, then read each region with a custom character-level OCR model
         and rebuild it as clean Markdown.</p>
      <span class="badge">DocLayout-YOLO</span>
      <span class="badge">Character-level OCR</span>
      <span class="badge">Markdown output</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# The page has rendered, clear the boot overlay.
boot_overlay.empty()


# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_hw():
    model = load_handwriting_model("handwriting_recognition_model.keras")
    classes = load_classes("classes.json")
    return model, classes


@st.cache_resource(show_spinner=False)
def get_doclayout():
    return load_doclayout_model()


# ---------------------------------------------------------------------------
# Geometry / layout helpers
# ---------------------------------------------------------------------------
def find_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_over_union(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = find_area(a) + find_area(b) - inter
    return inter / union if union > 0 else 0.0


def suppress_overlapping_blocks(blocks, iou_thresh=0.6):
    blocks = sorted(blocks, key=lambda b: b.score, reverse=True)
    saved = []
    for b in blocks:
        drop = False
        for k in saved:
            if intersection_over_union(b.coordinates, k.coordinates) >= iou_thresh:
                drop = True
                break
        if not drop:
            saved.append(b)

    saved.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
    return saved


def draw_layout_boxes(rgb_image, blocks):
    img = rgb_image.copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for b in blocks:
        x1, y1, x2, y2 = b.coordinates
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 80), 2)
        label = f"{b.type} ({b.score:.2f})"
        cv2.putText(
            bgr, label, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2, cv2.LINE_AA
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def warm_up_models(layout_mode):
    """Load models inside a visible, detailed loading screen."""
    with st.status("Warming up the models...", expanded=True) as status:
        st.write("Loading the handwriting recognition model...")
        t0 = time.time()
        hw_model, classes = get_hw()
        st.write(f"Handwriting model ready, {len(classes)} character classes, {time.time() - t0:.1f}s")

        doc_model = None
        if layout_mode == "DocLayout-YOLO":
            st.write(
                "Loading DocLayout-YOLO weights... "
                "_(first run downloads ~100 MB from Hugging Face and caches it)_"
            )
            t1 = time.time()
            doc_model = get_doclayout()
            st.write(f"Layout model ready, {time.time() - t1:.1f}s")

        status.update(label="Models ready", state="complete", expanded=False)
    return hw_model, classes, doc_model


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Options")

layout_mode = st.sidebar.selectbox(
    "Layout Mode",
    ["No layout", "DocLayout-YOLO"],
    index=1,
    help="DocLayout-YOLO splits the page into regions first; 'No layout' runs OCR on the whole image.",
)

doc_conf = st.sidebar.slider("Layout sensitivity", 0.05, 0.95, 0.20, 0.05)

st.sidebar.subheader("Handwriting Recognition")
min_contour_area = st.sidebar.slider("Min Contour Area", 10, 300, 50, 5)
line_threshold_factor = st.sidebar.slider("Line Spacing Sensitivity", 0.5, 3.0, 1.0, 0.1)
space_threshold_factor = st.sidebar.slider("Word Spacing Sensitivity", 0.1, 2.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: the first conversion is the slowest while the models load. "
    "Subsequent runs reuse the cached models."
)

# ---------------------------------------------------------------------------
# Main: upload + convert
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten notes image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload a `.jpg`, `.jpeg`, or `.png` of handwritten notes to get started.")
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input", anchor=False)
        st.image(rgb, use_container_width=True)

    if st.button("Convert", type="primary", use_container_width=True):
        # ---- Loading screen #1: models ----
        hw_model, classes, doc_model = warm_up_models(layout_mode)

        md_blocks = []
        layout_blocks = []
        char_debug = []
        final_text = ""

        # ---- Loading screen #2: the actual analysis ----
        with st.status("Analyzing your notes...", expanded=True) as status:
            if layout_mode == "DocLayout-YOLO":
                st.write("Detecting layout regions with DocLayout-YOLO...")
                raw_blocks = detect_layout_doclayout(
                    rgb, doc_model, conf=doc_conf, device="cpu", imgsz=1024
                )
                layout_blocks = suppress_overlapping_blocks(raw_blocks, iou_thresh=0.6)
                st.write(f"Found {len(layout_blocks)} region(s) after overlap filtering")

                total = max(1, len(layout_blocks))
                progress = st.progress(0.0, text="Reading handwriting...")

                for i, block in enumerate(layout_blocks):
                    progress.progress(
                        (i) / total,
                        text=f"Reading region {i + 1}/{len(layout_blocks)} - {block.type}",
                    )

                    region_rgb = crop_rgb_doc(rgb, block.coordinates)
                    if region_rgb.size == 0:
                        continue

                    region_bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)

                    text, dbg = ocr_char_level(
                        region_bgr, hw_model, classes,
                        min_contour_area=min_contour_area,
                        line_threshold_factor=line_threshold_factor,
                        space_threshold_factor=space_threshold_factor,
                        conf_threshold=0.0, return_debug=True,
                    )

                    text = text.strip()
                    if not text:
                        continue

                    label = block.type.lower()
                    if "title" in label:
                        md_blocks.append(f"# {text}")
                    elif "section" in label or "header" in label:
                        md_blocks.append(f"## {text}")
                    elif "list" in label:
                        for ln in text.splitlines():
                            ln = ln.strip()
                            if ln:
                                md_blocks.append(f"- {ln}")
                    elif "page-header" in label or "page-footer" in label:
                        continue
                    else:
                        md_blocks.append(text)

                    if dbg is not None:
                        char_debug.append({
                            "type": block.type,
                            "score": block.score,
                            "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                        })

                progress.progress(1.0, text="Done reading all regions")
                final_text = "\n\n".join(md_blocks) if md_blocks else ""

            else:
                st.write("Running OCR on the whole image...")
                final_text, dbg = ocr_char_level(
                    bgr, hw_model, classes,
                    min_contour_area=min_contour_area,
                    line_threshold_factor=line_threshold_factor,
                    space_threshold_factor=space_threshold_factor,
                    conf_threshold=0.0, return_debug=True,
                )
                char_debug = [{
                    "type": "Full image", "score": 1.0,
                    "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                }] if dbg is not None else []

            status.update(label="Analysis complete", state="complete", expanded=False)

        # ---- Output ----
        with col2:
            st.subheader("Output", anchor=False)
            if final_text.strip():
                tab_render, tab_source = st.tabs(["Rendered", "Markdown source"])
                with tab_render:
                    st.markdown(final_text)
                with tab_source:
                    st.code(final_text, language="markdown")
                    st.download_button(
                        "Download Markdown",
                        data=final_text,
                        file_name="notes.md",
                        mime="text/markdown",
                    )
            else:
                st.info("No text detected. Try adjusting the sensitivity sliders in the sidebar.")

        # ---- Visualizations ----
        st.markdown("---")
        st.subheader("Detection Visualizations", anchor=False)

        vis_col1, vis_col2 = st.columns([1, 1])

        with vis_col1:
            st.markdown("### Layout Detection")
            if layout_mode == "No layout":
                st.info("Layout mode is disabled.")
            else:
                st.caption(f"Blocks detected: {len(layout_blocks)} (after filtering)")
                layout_img = draw_layout_boxes(rgb, layout_blocks)
                st.image(layout_img, use_container_width=True)

        with vis_col2:
            st.markdown("### Character Detection (OCR)")
            if not char_debug:
                st.write("No character images to show.")
            else:
                for item in char_debug:
                    st.markdown(f"**{item['type']}** (score={item['score']:.2f})")
                    st.image(item["image"], use_container_width=True)

st.markdown(
    '<div class="footer">Built with DocLayout-YOLO and a custom Keras OCR model. '
    'Deployed on Streamlit Community Cloud.</div>',
    unsafe_allow_html=True,
)

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from doclayout_module import (
    load_doclayout_model,
    detect_layout_doclayout,
    crop_rgb as crop_rgb_doc,
)

from ocr_module import (
    load_handwriting_model,
    load_classes,
    ocr_char_level,
)

# ===============================
# Streamlit setup
# ===============================
st.set_page_config(page_title="Handwritten Notes → Structured Text", layout="wide")
st.title("Handwritten Notes → Structured Digital Text")

# ===============================
# Cached models
# ===============================
@st.cache_resource
def get_hw():
    model = load_handwriting_model("handwriting_recognition_model.keras")
    classes = load_classes("classes.json")
    return model, classes

@st.cache_resource
def get_doclayout():
    return load_doclayout_model()


# ===============================
# Helpers: layout postprocessing + drawing
# ===============================
def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0

def _contains(outer, inner, tol=0):
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (ix1 >= ox1 - tol) and (iy1 >= oy1 - tol) and (ix2 <= ox2 + tol) and (iy2 <= oy2 + tol)

def suppress_overlapping_blocks(blocks, iou_thresh=0.6, containment_tol=8):
    """
    Removes duplicates/overlaps so you don't see "layered blocks".
    Strategy:
      - sort by score desc (keep strongest)
      - drop block if highly overlapping with a kept one
      - or if it's fully contained by a kept one with same/very similar label
    """
    # sort high confidence first
    blocks = sorted(blocks, key=lambda b: b.score, reverse=True)

    kept = []
    for b in blocks:
        drop = False
        for k in kept:
            iou = _iou(b.coordinates, k.coordinates)

            # 1) strong overlap -> drop weaker
            if iou >= iou_thresh:
                drop = True
                break

            # 2) if contained and same-ish type, drop inner duplicate
            # (sometimes model outputs title+title inside each other)
            if _contains(k.coordinates, b.coordinates, tol=containment_tol):
                if b.type.lower() == k.type.lower():
                    drop = True
                    break

        if not drop:
            kept.append(b)

    # reading order sort
    kept.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
    return kept

def draw_layout_boxes(rgb_image, blocks):
    """
    Draw layout blocks on RGB image. Returns RGB image.
    """
    img = rgb_image.copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for b in blocks:
        x1, y1, x2, y2 = b.coordinates
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{b.type} ({b.score:.2f})"
        cv2.putText(
            bgr, label, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ===============================
# Sidebar: simplified
# ===============================
st.sidebar.header("Options")

layout_mode = st.sidebar.selectbox(
    "Layout mode",
    ["No layout", "DocLayout-YOLO (pretrained)"],
    index=1
)

doc_conf = st.sidebar.slider("Layout sensitivity", 0.05, 0.95, 0.20, 0.05)
device = st.sidebar.selectbox("Device", ["cpu"], index=0)

# Simple toggles for normal users
show_layout_view = st.sidebar.checkbox("Show layout view", value=True)
show_char_view = st.sidebar.checkbox("Show character view", value=True)

# Advanced settings tucked away
with st.sidebar.expander("Advanced settings"):
    # Layout suppression sliders
    iou_thresh = st.slider("Layout overlap filter (IoU)", 0.3, 0.9, 0.6, 0.05)
    containment_tol = st.slider("Containment tolerance (px)", 0, 30, 8, 1)

    # OCR knobs
    min_contour_area = st.slider("min_contour_area", 10, 300, 50, 5)
    line_threshold_factor = st.slider("line_threshold_factor", 0.5, 3.0, 1.0, 0.1)
    space_threshold_factor = st.slider("space_threshold_factor", 0.1, 2.0, 0.4, 0.05)
    conf_threshold = st.slider("character confidence threshold", 0.0, 1.0, 0.0, 0.05)

uploaded_file = st.file_uploader("Upload a handwritten notes image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        st.image(rgb, use_container_width=True)

    if st.button("Convert to Structured Text"):
        hw_model, classes = get_hw()

        md_blocks = []
        char_debug = []   # list of dicts with block label + char overlay
        layout_blocks = []

        # ===============================
        # Layout path
        # ===============================
        if layout_mode == "DocLayout-YOLO (pretrained)":
            doc_model = get_doclayout()

            raw_blocks = detect_layout_doclayout(
                rgb,
                doc_model,
                conf=doc_conf,
                device=device,
                imgsz=1024
            )

            # Remove duplicates/overlaps so "layered blocks" go away
            layout_blocks = suppress_overlapping_blocks(
                raw_blocks,
                iou_thresh=iou_thresh,
                containment_tol=containment_tol
            )

            # Build text by blocks
            for block in layout_blocks:
                region_rgb = crop_rgb_doc(rgb, block.coordinates)
                if region_rgb.size == 0:
                    continue

                region_bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)

                text, dbg = ocr_char_level(
                    region_bgr,
                    hw_model,
                    classes,
                    min_contour_area=min_contour_area,
                    line_threshold_factor=line_threshold_factor,
                    space_threshold_factor=space_threshold_factor,
                    conf_threshold=conf_threshold,
                    return_debug=True
                )

                text = text.strip()
                if not text:
                    continue

                label = block.type.lower()

                # Layout → Markdown mapping
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

                if show_char_view and dbg is not None:
                    char_debug.append({
                        "type": block.type,
                        "score": block.score,
                        "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                    })

            final_text = "\n\n".join(md_blocks) if md_blocks else ""

        # ===============================
        # No-layout fallback
        # ===============================
        else:
            final_text, dbg = ocr_char_level(
                bgr,
                hw_model,
                classes,
                min_contour_area=min_contour_area,
                line_threshold_factor=line_threshold_factor,
                space_threshold_factor=space_threshold_factor,
                conf_threshold=conf_threshold,
                return_debug=True
            )

            layout_blocks = []
            if show_char_view and dbg is not None:
                char_debug = [{
                    "type": "Full image",
                    "score": 1.0,
                    "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                }]

        # ===============================
        # Output UI (separated views)
        # ===============================
        # ===============================
        # ROW 1 — Output text (unchanged)
        # ===============================
        with col2:
            st.subheader("Output")
            if final_text.strip():
                st.code(final_text, language="markdown")
            else:
                st.info("No text detected. Try lowering layout sensitivity or adjusting Advanced OCR settings.")

        # ===============================
        # ROW 2 — Visualizations
        # ===============================
        if show_layout_view or show_char_view:
            st.markdown("---")
            st.subheader("Detection Visualizations")

            vis_col1, vis_col2 = st.columns([1, 1])

            # ---- LEFT: Layout detection ----
            with vis_col1:
                if show_layout_view and layout_mode != "No layout":
                    st.markdown("### Layout detection (blocks)")
                    st.caption(f"Blocks detected: {len(layout_blocks)} (after overlap filtering)")
                    layout_img = draw_layout_boxes(rgb, layout_blocks)
                    st.image(layout_img, use_container_width=True)

                    with st.expander("Show block list"):
                        for i, b in enumerate(layout_blocks, start=1):
                            st.write(f"{i}. **{b.type}** score={b.score:.2f} box={b.coordinates}")
                else:
                    st.info("Layout view disabled.")

            # ---- RIGHT: Character detection ----
            with vis_col2:
                if show_char_view:
                    st.markdown("### Character detection (OCR)")
                    if not char_debug:
                        st.write("No character debug images to show.")
                    else:
                        for item in char_debug:
                            st.markdown(f"**{item['type']}** (score={item['score']:.2f})")
                            st.image(item["image"], use_container_width=True)
                else:
                    st.info("Character view disabled.")


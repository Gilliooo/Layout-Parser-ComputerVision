import streamlit as st
import numpy as np
import cv2
from PIL import Image

from doclayout_module import (load_doclayout_model, detect_layout_doclayout, crop_rgb as crop_rgb_doc,
)

from ocr_module import (load_handwriting_model,load_classes, ocr_char_level,
)

# setting up streamlit page
st.set_page_config(page_title="Handwritten Notes to Structured Text", layout="wide")
st.title("Handwritten Notes to Structured Text")

# streamlit loading model and using cache
@st.cache_resource
def get_hw():
    model = load_handwriting_model("handwriting_recognition_model.keras")
    classes = load_classes("classes.json")
    return model, classes

@st.cache_resource
def get_doclayout():
    return load_doclayout_model()

# area computing
def find_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

# intersection over union to filter areas
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

# filtering for overlaps
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

# drawing the box in the image for visualization
def draw_layout_boxes(rgb_image, blocks):
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

# sidebar
st.sidebar.header("Options")

layout_mode = st.sidebar.selectbox(
    "Layout Mode",
    ["No layout", "DocLayout-YOLO"],
    index=1
)

# layout sensitivity
doc_conf = st.sidebar.slider("Layout sensitivity", 0.05, 0.95, 0.20, 0.05)

# controls for handwriting recognition
st.sidebar.subheader("Handwriting Recognition Settings")
min_contour_area = st.sidebar.slider("Min Contour Area", 10, 300, 50, 5)
line_threshold_factor = st.sidebar.slider("Line Spacing Sensitivity", 0.5, 3.0, 1.0, 0.1)
space_threshold_factor = st.sidebar.slider("Word Spacing Sensitivity", 0.1, 2.0, 0.4, 0.05)

uploaded_file = st.file_uploader("Upload a handwritten notes image!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # input output row
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        st.image(rgb, use_container_width=True)

    if st.button("Convert"):
        hw_model, classes = get_hw()

        md_blocks = []
        layout_blocks = []
        char_debug = []

        if layout_mode == "DocLayout-YOLO":
            doc_model = get_doclayout()

            raw_blocks = detect_layout_doclayout(
                rgb,
                doc_model,
                conf=doc_conf,
                device="cpu",
                imgsz=1024
            )

            layout_blocks = suppress_overlapping_blocks(raw_blocks, iou_thresh=0.6)

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
                    conf_threshold=0.0,
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

                if dbg is not None:
                    char_debug.append({
                        "type": block.type,
                        "score": block.score,
                        "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                    })

            final_text = "\n\n".join(md_blocks) if md_blocks else ""

        else:
            final_text, dbg = ocr_char_level(
                bgr,
                hw_model,
                classes,
                min_contour_area=min_contour_area,
                line_threshold_factor=line_threshold_factor,
                space_threshold_factor=space_threshold_factor,
                conf_threshold=0.0,
                return_debug=True
            )

            char_debug = [{
                "type": "Full image",
                "score": 1.0,
                "image": cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
            }] if dbg is not None else []

        # ===============================
        # Output (Row 1 right)
        # ===============================
        with col2:
            st.subheader("Output")
            if final_text.strip():
                st.code(final_text, language="markdown")
            else:
                st.info("No text detected.")

        st.markdown("---")
        st.subheader("Detection Visualizations")

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

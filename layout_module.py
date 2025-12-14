# layout_module.py
import layoutparser as lp

def load_layout_model(score_thresh=0.8):
    return lp.Detectron2LayoutModel(
        "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

def filter_and_sort_blocks(layout, keep_types=("Title", "Text", "List"), remove_figures=True):
    blocks = [b for b in layout if b.type in keep_types]

    if remove_figures:
        figure_blocks = [b for b in layout if b.type == "Figure"]
        blocks = [b for b in blocks if not any(b.is_in(fb) for fb in figure_blocks)]

    # top-to-bottom
    blocks.sort(key=lambda b: b.coordinates[1])
    return blocks

def crop_rgb(rgb_image, block):
    x1, y1, x2, y2 = map(int, block.coordinates)
    h, w = rgb_image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return rgb_image[y1:y2, x1:x2]

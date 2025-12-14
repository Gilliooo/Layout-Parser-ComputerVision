# ocr_module.py
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 32

def load_classes(path="classes.json"):
    with open(path, "r") as f:
        return np.array(json.load(f), dtype=object)

def load_handwriting_model(path="handwriting_recognition_model.keras"):
    return load_model(path)

def _predict_char(model, classes, roi_gray):
    roi = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    x = roi.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)      # (32,32,1)
    x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return str(classes[idx]), float(probs[idx])

def ocr_char_level(
    bgr_image,
    model,
    classes,
    min_contour_area=50,
    line_threshold_factor=1.0,
    space_threshold_factor=0.4,
    conf_threshold=0.0,
    return_debug=False
):
    """
    Contour-based character segmentation + classifier.
    Returns:
      - text (str)
      - (text, debug_img_bgr) if return_debug=True
    """

    if bgr_image is None or bgr_image.size == 0:
        return ("", None) if return_debug else ""

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    global_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    debug_img = cv2.cvtColor(global_thresh, cv2.COLOR_GRAY2BGR)
    dilated = cv2.dilate(global_thresh.copy(), None, iterations=2)

    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for c in cnts:
        if cv2.contourArea(c) > min_contour_area:
            x, y, w, h = cv2.boundingRect(c)
            chars.append({"bbox": (x, y, w, h)})

    if not chars:
        return ("", debug_img) if return_debug else ""

    chars.sort(key=lambda item: item["bbox"][1])  # sort by y

    heights = [c["bbox"][3] for c in chars]
    widths  = [c["bbox"][2] for c in chars]
    avg_h = float(np.mean(heights)) if heights else 0.0
    avg_w = float(np.mean(widths)) if widths else 0.0

    # group into lines
    lines = []
    current = [chars[0]]
    for i in range(1, len(chars)):
        prev_y = chars[i-1]["bbox"][1]
        cur_y  = chars[i]["bbox"][1]
        if (cur_y - prev_y) > (avg_h * line_threshold_factor):
            lines.append(current)
            current = [chars[i]]
        else:
            current.append(chars[i])
    lines.append(current)

    out_lines = []

    for line in lines:
        line = sorted(line, key=lambda item: item["bbox"][0])  # left to right

        line_chars = []
        prev_end_x = None

        for item in line:
            x, y, w, h = item["bbox"]

            if prev_end_x is not None:
                gap = x - prev_end_x
                if avg_w > 0 and gap > (avg_w * space_threshold_factor):
                    line_chars.append(" ")

            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            pred, conf = _predict_char(model, classes, roi)
            if conf >= conf_threshold:
                line_chars.append(pred)

            if return_debug:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            prev_end_x = x + w

        out_lines.append("".join(line_chars).strip())

    text = "\n".join([ln for ln in out_lines if ln])
    return (text, debug_img) if return_debug else text

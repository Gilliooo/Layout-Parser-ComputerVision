# doclayout_module.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10


@dataclass
class Block:
    type: str
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float


def load_doclayout_model(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
):

    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )
    model = YOLOv10(weights_path)
    return model


def detect_layout_doclayout(
    rgb_image: np.ndarray,
    model,
    conf: float = 0.2,
    imgsz: int = 1024,
    device: str = "cpu",
) -> List[Block]:

    results = model.predict(
        rgb_image,
        conf=conf,
        imgsz=imgsz,
        device=device
    )

    r = results[0]
    blocks: List[Block] = []

    if r.boxes is None:
        return blocks

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    names = r.names

    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
        blocks.append(
            Block(
                type=str(names[c]),
                coordinates=(int(x1), int(y1), int(x2), int(y2)),
                score=float(s)
            )
        )

    # reading order
    blocks.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
    return blocks


def crop_rgb(rgb_image: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = coords
    h, w = rgb_image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return rgb_image[y1:y2, x1:x2]

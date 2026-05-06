from pathlib import Path
from typing import Iterable
import cv2


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_box(img, box, color=(0, 255, 0), thickness=2, label=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    if label:
        y = max(15, y1 - 8)
        cv2.putText(
            out,
            str(label),
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_many_boxes(img, items: Iterable[dict]):
    out = img.copy()
    for item in items:
        box = item["box"]
        color = item.get("color", (0, 255, 0))
        label = item.get("label")
        thickness = item.get("thickness", 2)
        out = draw_box(out, box, color=color, thickness=thickness, label=label)
    return out


def save_image(path: Path, img):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def crop_box(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()
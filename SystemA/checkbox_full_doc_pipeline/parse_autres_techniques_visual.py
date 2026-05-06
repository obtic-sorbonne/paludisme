from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np


def clean_text(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_accents_basic(s: str) -> str:
    repl = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "ù": "u", "û": "u",
        "ï": "i", "î": "i",
        "ô": "o", "ö": "o",
        "ç": "c",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "À": "A", "Â": "A",
        "Ù": "U", "Û": "U",
        "Ï": "I", "Î": "I",
        "Ô": "O", "Ö": "O",
        "Ç": "C",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def norm(s: str) -> str:
    s = clean_text(s)
    s = s.replace("：", ":").replace("’", "'").replace("−", "-")
    s = strip_accents_basic(s).lower()
    return clean_text(s)


def compact_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_page_words(data: dict, page_num: int = 1):
    out = []

    def add_word(text, box):
        if not text or box is None:
            return

        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
        elif len(box) == 4 and isinstance(box[0], (list, tuple)):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            return

        txt = clean_text(text)
        if not txt:
            return

        out.append({
            "text": txt,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": (float(x1) + float(x2)) / 2.0,
            "cy": (float(y1) + float(y2)) / 2.0,
            "w": float(x2) - float(x1),
            "h": float(y2) - float(y1),
        })

    if isinstance(data, dict) and "overall_ocr_res" in data:
        ocr = data["overall_ocr_res"]
        if isinstance(ocr, dict) and "rec_texts" in ocr and "dt_polys" in ocr:
            for txt, box in zip(ocr["rec_texts"], ocr["dt_polys"]):
                add_word(txt, box)
            return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    if isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            txt = w.get("text") or w.get("word_text") or w.get("transcription")
            box = w.get("box") or w.get("bbox") or w.get("points")
            add_word(txt, box)

    return sorted(out, key=lambda w: (w["y1"], w["x1"]))


def text_matches(word_text: str, variants: list[str]) -> bool:
    wt = norm(word_text)
    wc = compact_norm(word_text)

    for v in variants:
        vn = norm(v)
        vc = compact_norm(v)

        if wt == vn or wc == vc:
            return True
        if vc and vc in wc:
            return True

    return False


def crop_box(img_bgr, box):
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img_bgr[y1:y2, x1:x2].copy()


def marker_box_for_option_word(word, label):
    # wider crop for the first three options because their checkbox is farther left
    if label in {"PCR", "QBC", "Sérologie"}:
        left_width = 72
        right_gap = 8
        y_pad = 10
    else:
        left_width = 42
        right_gap = 5
        y_pad = 8

    return [
        int(word["x1"] - left_width),
        int(word["y1"] - y_pad),
        int(word["x1"] - right_gap),
        int(word["y2"] + y_pad),
    ]


def score_square_checkbox(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "state": "unknown",
            "dark_ratio": 0.0,
            "largest_area": 0,
            "component_count": 0,
        }

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    h, w = th.shape[:2]
    total = float(max(1, h * w))
    dark_ratio = float(np.count_nonzero(th)) / total

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 3:
            areas.append(area)

    largest_area = max(areas) if areas else 0
    component_count = len(areas)

    if dark_ratio >= 0.09 or largest_area >= 20:
        state = "selected"
    elif dark_ratio >= 0.012 or largest_area >= 6:
        state = "unselected"
    else:
        state = "unknown"

    return {
        "state": state,
        "dark_ratio": round(dark_ratio, 4),
        "largest_area": largest_area,
        "component_count": component_count,
    }


def parse_autres_techniques_visual(words, img_bgr):
    result = {
        "field": "Autres techniques visuel",
        "found": False,
        "selected_options": [],
        "found_options": [],
        "options": [],
    }

    anchor = None
    for w in words:
        if text_matches(w["text"], ["Autres techniques"]):
            anchor = w
            break

    if not anchor:
        return result

    result["found"] = True

    local_words = []
    for w in words:
        if (
            w["x1"] >= anchor["x1"] - 80
            and w["x2"] <= anchor["x1"] + 1300
            and w["y1"] >= anchor["y1"] + 5
            and w["y2"] <= anchor["y1"] + 130
        ):
            local_words.append(w)

    options = [
        ("PCR", ["PCR"]),
        ("QBC", ["QBC"]),
        ("Sérologie", ["Sérologie", "Serologie"]),
        ("Autre", ["Autre"]),
    ]

    for label, variants in options:
        candidates = []

        for w in local_words:
            txt_n = norm(w["text"])
            comp = compact_norm(w["text"])

            if label == "Autre" and "autrestechniques" in comp:
                continue

            if text_matches(w["text"], variants):
                candidates.append(w)
                continue

            for v in variants:
                vc = compact_norm(v)
                if vc and vc in comp:
                    candidates.append(w)
                    break

        if not candidates:
            result["options"].append({
                "label": label,
                "found_word": False,
                "state": "missing",
            })
            continue

        word = sorted(
            candidates,
            key=lambda w: (abs(w["cy"] - (anchor["cy"] + 45)), w["x1"])
        )[0]

        marker_box = marker_box_for_option_word(word, label)
        crop = crop_box(img_bgr, marker_box)
        score = score_square_checkbox(crop)

        if label not in result["found_options"]:
            result["found_options"].append(label)

        if score["state"] == "selected" and label not in result["selected_options"]:
            result["selected_options"].append(label)

        result["options"].append({
            "label": label,
            "found_word": True,
            "word_text": word["text"],
            "word_box": [word["x1"], word["y1"], word["x2"], word["y2"]],
            "marker_box": marker_box,
            "state": score["state"],
            "dark_ratio": score["dark_ratio"],
            "largest_area": score["largest_area"],
            "component_count": score["component_count"],
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse Autres techniques visually")
    parser.add_argument("ocr_json_path")
    parser.add_argument("page_image_path")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/autres_techniques_visual",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    page_image_path = Path(args.page_image_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(ocr_json_path)
    data = data.get("res", data)
    words = extract_page_words(data, page_num=args.page_num)

    img_bgr = cv2.imread(str(page_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {page_image_path}")

    result = parse_autres_techniques_visual(words, img_bgr)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(page_image_path),
        "page_num": args.page_num,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_autres_techniques_visual.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
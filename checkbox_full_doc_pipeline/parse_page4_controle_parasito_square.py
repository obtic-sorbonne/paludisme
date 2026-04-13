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
    return re.sub(r"[^a-z0-9><=+/\-]", "", norm(s))


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

    if isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            txt = w.get("text") or w.get("word_text") or w.get("transcription")
            box = w.get("box") or w.get("bbox") or w.get("points")
            add_word(txt, box)
        return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    if isinstance(data, dict) and "overall_ocr_res" in data:
        ocr = data["overall_ocr_res"]

        if isinstance(ocr, dict):
            if "rec_texts" in ocr and "dt_polys" in ocr:
                for txt, box in zip(ocr["rec_texts"], ocr["dt_polys"]):
                    add_word(txt, box)
            elif "ocr_results" in ocr and isinstance(ocr["ocr_results"], list):
                for item in ocr["ocr_results"]:
                    if isinstance(item, dict):
                        txt = item.get("text") or item.get("word_text") or item.get("transcription")
                        box = item.get("box") or item.get("bbox") or item.get("points") or item.get("poly")
                        add_word(txt, box)

        elif isinstance(ocr, list):
            for item in ocr:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("word_text") or item.get("transcription")
                    box = item.get("box") or item.get("bbox") or item.get("points") or item.get("poly")
                    add_word(txt, box)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    box = item[0]
                    text_info = item[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        txt = text_info[0]
                        add_word(txt, box)

        return sorted(out, key=lambda w: (w["y1"], w["x1"]))

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


def find_best_word(words, variants):
    candidates = [w for w in words if text_matches(w["text"], variants)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda w: (w["y1"], w["x1"]))[0]


def window_words(words, x1=None, y1=None, x2=None, y2=None):
    out = []
    for w in words:
        if x1 is not None and w["x2"] < x1:
            continue
        if x2 is not None and w["x1"] > x2:
            continue
        if y1 is not None and w["y2"] < y1:
            continue
        if y2 is not None and w["y1"] > y2:
            continue
        out.append(w)
    return out


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
    _, th = cv2.threshold(blur, 185, 255, cv2.THRESH_BINARY_INV)

    h, w = th.shape[:2]
    if h > 6 and w > 6:
        inner = th[2:h - 2, 2:w - 2]
    else:
        inner = th

    total = float(max(1, inner.shape[0] * inner.shape[1]))
    dark_ratio = float(np.count_nonzero(inner)) / total

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 3:
            areas.append(area)

    largest_area = max(areas) if areas else 0
    component_count = len(areas)

    if dark_ratio >= 0.06 or largest_area >= 14:
        state = "selected"
    elif dark_ratio >= 0.012 or largest_area >= 5:
        state = "unselected"
    else:
        state = "unknown"

    return {
        "state": state,
        "dark_ratio": round(dark_ratio, 4),
        "largest_area": largest_area,
        "component_count": component_count,
    }


def build_left_checkbox_box(option_word, x_gap_max=42, y_pad=8):
    return [
        option_word["x1"] - x_gap_max,
        option_word["y1"] - y_pad,
        option_word["x1"] - 4,
        option_word["y2"] + y_pad,
    ]


def checkbox_state_for_word(img_bgr, option_word, x_gap_max=42, y_pad=8):
    crop = crop_box(img_bgr, build_left_checkbox_box(option_word, x_gap_max=x_gap_max, y_pad=y_pad))
    return score_square_checkbox(crop)


def closest_numeric_temperature(row_words, row_anchor, x_min=None, x_max=None, max_row_dist=20):
    candidates = []

    for w in row_words:
        if x_min is not None and w["x1"] < x_min:
            continue
        if x_max is not None and w["x2"] > x_max:
            continue

        txt = w["text"].replace(",", ".")
        if re.fullmatch(r"\d{1,2}\.\d", txt):
            dist = abs(w["cy"] - row_anchor["cy"])
            if dist <= max_row_dist:
                candidates.append((dist, abs(w["cx"] - row_anchor["cx"]), w))

    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: (x[0], x[1]))[0][2]
    return best["text"].replace(".", ",")


def extract_comment_line(words, anchor_y):
    comment_words = [
        w for w in words
        if w["y1"] >= anchor_y + 18
    ]
    if not comment_words:
        return None

    line_words = sorted(comment_words, key=lambda w: (w["y1"], w["x1"]))
    texts = []
    for w in line_words:
        txt = clean_text(w["text"])
        if txt:
            texts.append(txt)

    if not texts:
        return None

    return " ".join(texts)


def parse_controle_block(words, img_bgr):
    result = {
        "field": "Contrôle parasitologique P falciparum",
        "found": False,
        "control_overall": None,
        "rows": [],
        "commentaires_remarques": None,
        "perdu_de_vue": None,
        "template_type": "square",
    }

    control_anchor = find_best_word(
        words,
        ["Contrôle parasitologique P falciparum", "Controle parasitologique P falciparum"]
    )
    if not control_anchor:
        return result

    result["found"] = True

    row_defs = [
        ("J3 ou J4", ["J3 ou J4"]),
        ("J7 +/-1", ["J7 +/-1", "J7 +/- 1"]),
        ("J28 +/-2", ["J28 +/-2", "J28 +/- 2", "J28+/-2"]),
        ("Autre", ["Autre"]),
    ]

    paras_specs = [
        ("Absence", ["absence"]),
        ("Trophos", ["trophos"]),
        ("Gaméto seuls", ["gaméto seuls", "gameto seuls", "gametoseuls"]),
    ]

    dens_specs = [
        ("≤ 100", ["≤100", "≤ 100", "<=100", "≤100"]),
        ("101-10 000", ["101-10 000", "101-10000"]),
        ("> 10 000", ["> 10 000", ">10000"]),
    ]

    table_zone = window_words(
        words,
        x1=control_anchor["x1"] - 20,
        x2=control_anchor["x1"] + 1100,
        y1=control_anchor["y2"] - 5,
        y2=control_anchor["y2"] + 210,
    )

    rows_found = []
    for label, variants in row_defs:
        w = find_best_word(table_zone, variants)
        if w:
            rows_found.append((label, w))

    expected_order = ["J3 ou J4", "J7 +/-1", "J28 +/-2", "Autre"]
    rank = {name: i for i, name in enumerate(expected_order)}
    rows_found = sorted(rows_found, key=lambda x: (rank.get(x[0], 999), x[1]["cy"]))

    temp_anchor = find_best_word(words, ["Température", "Temperature"])
    paras_anchor = find_best_word(words, ["Parasitologie"])
    dens_anchor = find_best_word(words, ["Densité parasitaire", "Densite parasitaire"])

    temp_x_min = temp_anchor["x1"] - 30 if temp_anchor else None
    temp_x_max = temp_anchor["x2"] + 70 if temp_anchor else None

    paras_x_min = paras_anchor["x1"] - 10 if paras_anchor else None
    paras_x_max = paras_anchor["x2"] + 330 if paras_anchor else None

    dens_x_min = dens_anchor["x1"] - 10 if dens_anchor else None
    dens_x_max = dens_anchor["x2"] + 320 if dens_anchor else None

    for idx, (row_label, row_anchor) in enumerate(rows_found):
        if idx == 0:
            row_top = row_anchor["cy"] - 14
        else:
            prev_anchor = rows_found[idx - 1][1]
            row_top = (prev_anchor["cy"] + row_anchor["cy"]) / 2.0

        if idx + 1 < len(rows_found):
            next_anchor = rows_found[idx + 1][1]
            row_bottom = (row_anchor["cy"] + next_anchor["cy"]) / 2.0
        else:
            row_bottom = row_anchor["cy"] + 16

        row_words = window_words(words, y1=row_top, y2=row_bottom)

        row = {
            "row": row_label,
            "fait": None,
            "temperature": None,
            "parasitologie": [],
            "densite_parasitaire": [],
        }

        # Fait checkbox = immediately right of row label
        fait_box = [
            row_anchor["x2"] + 5,
            row_anchor["y1"] - 6,
            row_anchor["x2"] + 30,
            row_anchor["y2"] + 6,
        ]
        fait_crop = crop_box(img_bgr, fait_box)
        fait_score = score_square_checkbox(fait_crop)
        if fait_score["state"] == "selected":
            row["fait"] = "Oui"

        if temp_x_min is not None and temp_x_max is not None:
            temp_words = window_words(row_words, x1=temp_x_min, x2=temp_x_max)
            row["temperature"] = closest_numeric_temperature(
                temp_words,
                row_anchor,
                x_min=temp_x_min,
                x_max=temp_x_max,
                max_row_dist=16,
            )

        if paras_x_min is not None and paras_x_max is not None:
            paras_words = window_words(row_words, x1=paras_x_min, x2=paras_x_max)
        else:
            paras_words = row_words

        for canonical, variants in paras_specs:
            option_word = find_best_word(paras_words, variants)
            if not option_word:
                continue
            score = checkbox_state_for_word(img_bgr, option_word, x_gap_max=28, y_pad=8)
            if score["state"] == "selected":
                row["parasitologie"].append(canonical)

        if dens_x_min is not None and dens_x_max is not None:
            dens_words = window_words(row_words, x1=dens_x_min, x2=dens_x_max)
        else:
            dens_words = row_words

        for canonical, variants in dens_specs:
            option_word = find_best_word(dens_words, variants)
            if not option_word:
                continue
            score = checkbox_state_for_word(img_bgr, option_word, x_gap_max=28, y_pad=8)
            if score["state"] == "selected":
                row["densite_parasitaire"].append(canonical)

        if row["fait"] is None:
            if row["temperature"] or row["parasitologie"] or row["densite_parasitaire"]:
                row["fait"] = "Oui"

        result["rows"].append(row)

    # comment block
    comment_anchor = find_best_word(words, ["Commentaires & remarques", "Commentaires", "Remarques"])
    if comment_anchor:
        comment_words = window_words(
            words,
            x1=comment_anchor["x1"] - 20,
            x2=comment_anchor["x1"] + 900,
            y1=comment_anchor["y2"] + 5,
            y2=comment_anchor["y2"] + 120,
        )
        if comment_words:
            texts = [clean_text(w["text"]) for w in sorted(comment_words, key=lambda z: (z["y1"], z["x1"])) if clean_text(w["text"])]
            if texts:
                result["commentaires_remarques"] = " ".join(texts)

    # keep this as None for now in square template until we add a dedicated rule
    result["perdu_de_vue"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse square-template contrôle parasitologique block")
    parser.add_argument("ocr_json_path", help="Path to OCR JSON with word boxes")
    parser.add_argument("--page-image-path", required=True, help="Page image path")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page4_controle_parasito_square",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(ocr_json_path)
    data = data.get("res", data)
    words = extract_page_words(data, page_num=args.page_num)

    img_bgr = cv2.imread(str(args.page_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.page_image_path}")

    result = parse_controle_block(words, img_bgr=img_bgr)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(args.page_image_path),
        "page_num": args.page_num,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_controle_parasito_square.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded:       {len(words)}")
    print(f"Saved JSON:         {out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
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
    return re.sub(r"[^a-z0-9><=+/-]", "", norm(s))


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


def find_word_near(words, variants, x1=None, x2=None, y1=None, y2=None):
    cands = []
    for w in words:
        if x1 is not None and w["x2"] < x1:
            continue
        if x2 is not None and w["x1"] > x2:
            continue
        if y1 is not None and w["y2"] < y1:
            continue
        if y2 is not None and w["y1"] > y2:
            continue
        if text_matches(w["text"], variants):
            cands.append(w)

    if not cands:
        return None
    return sorted(cands, key=lambda w: (w["y1"], w["x1"]))[0]


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


def token_is_selected_marker_only(w) -> bool:
    s = clean_text(w["text"])
    return s in {"X", "x", "×", "☒", "☑", "■", "█", "▪", "▣"}


def token_is_unselected_marker_only(w) -> bool:
    s = clean_text(w["text"])
    return s in {"O", "o", "□", "○", "◯", "◻", "◽"}


def merged_token_selection_for_variants(word_text: str, variants: list[str]):
    raw = clean_text(word_text)
    n = norm(raw)

    for variant in variants:
        vn = norm(variant)
        if not vn:
            continue

        if n.endswith(vn) and n != vn:
            prefix = n[:len(n) - len(vn)].strip()
            if prefix in {"x", "×"}:
                return "X"
            if prefix in {"o", "□", "○", "◯"}:
                return "O"

        cn = compact_norm(raw)
        cv = compact_norm(variant)
        if cv and cn.endswith(cv) and len(cn) > len(cv):
            prefix = cn[:len(cn) - len(cv)]
            if prefix == "x":
                return "X"
            if prefix == "o":
                return "O"

    return None


def find_left_marker(row_words, option_word, x_gap_max=65, y_pad=12):
    left = window_words(
        row_words,
        x1=option_word["x1"] - x_gap_max,
        x2=option_word["x1"] - 2,
        y1=option_word["y1"] - y_pad,
        y2=option_word["y2"] + y_pad,
    )

    left_sorted = sorted(left, key=lambda w: abs(w["cx"] - option_word["x1"]))
    for w in left_sorted:
        if token_is_selected_marker_only(w):
            return "X"
        if token_is_unselected_marker_only(w):
            return "O"
    return None


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


def build_left_checkbox_box(option_word, x_gap_max=120, y_pad=22):
    return [
        option_word["x1"] - x_gap_max,
        option_word["y1"] - y_pad,
        option_word["x1"] - 2,
        option_word["y2"] + y_pad,
    ]


def score_checkbox_crop(crop_bgr):
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


def detect_checkbox_or_marker_selected(
    row_words,
    option_word,
    variants,
    img_bgr=None,
    x_gap_max=110,
    y_pad=22,
):
    marker = find_left_marker(row_words, option_word, x_gap_max=x_gap_max, y_pad=y_pad)
    if marker == "X":
        return True
    if marker == "O":
        return False

    merged = merged_token_selection_for_variants(option_word["text"], variants)
    if merged == "X":
        return True
    if merged == "O":
        return False

    if img_bgr is not None:
        crop = crop_box(img_bgr, build_left_checkbox_box(option_word, x_gap_max=x_gap_max, y_pad=y_pad))
        score = score_checkbox_crop(crop)
        if score["state"] == "selected":
            return True
        if score["state"] == "unselected":
            return False

    return None


def closest_numeric_temperature(row_words, row_anchor, next_row_cy=None, x_min=None, x_max=None, max_row_dist=22):
    candidates = []

    for w in row_words:
        if x_min is not None and w["x1"] < x_min:
            continue
        if x_max is not None and w["x2"] > x_max:
            continue

        txt = w["text"].replace(",", ".")
        if re.fullmatch(r"\d{1,2}\.\d", txt):
            dist_cur = abs(w["cy"] - row_anchor["cy"])
            if dist_cur > max_row_dist:
                continue

            dist_next = abs(w["cy"] - next_row_cy) if next_row_cy is not None else float("inf")
            if dist_cur < dist_next:
                candidates.append((dist_cur, abs(w["cx"] - row_anchor["cx"]), w))

    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: (x[0], x[1]))[0][2]
    return best["text"].replace(".", ",")


def parse_controle_block(words, img_bgr=None):
    result = {
        "field": "Contrôle parasitologique P falciparum",
        "found": False,
        "control_overall": None,
        "rows": [],
        "commentaires_remarques": None,
        "perdu_de_vue": None,
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
        ("Absence", ["absence", "bsence"]),
        ("Trophos", ["trophos"]),
        ("Gaméto seuls", ["gaméto seuls", "gameto seuls", "gamétoseuls", "gametoseuls"]),
    ]

    dens_specs = [
        ("≤ 100", ["≤100", "≤ 100", "<=100"]),
        ("101-10 000", ["101-10 000", "101-10000"]),
        ("> 10 000", ["> 10 000", ">10000"]),
    ]

    table_zone = window_words(
        words,
        x1=control_anchor["x1"] - 40,
        x2=control_anchor["x1"] + 1040,
        y1=control_anchor["y2"] - 5,
        y2=control_anchor["y2"] + 340,
    )

    rows_found = []
    for label, variants in row_defs:
        w = find_best_word(table_zone, variants)
        if w:
            rows_found.append((label, w))

    if not rows_found:
        return result

    expected_order = ["J3 ou J4", "J7 +/-1", "J28 +/-2", "Autre"]
    order_rank = {name: i for i, name in enumerate(expected_order)}
    rows_found = sorted(rows_found, key=lambda x: (order_rank.get(x[0], 999), x[1]["cy"]))

    near_heading = window_words(
        words,
        x1=control_anchor["x1"] - 10,
        x2=control_anchor["x1"] + 420,
        y1=control_anchor["y1"] - 10,
        y2=control_anchor["y2"] + 45,
    )

    for w in near_heading:
        if text_matches(w["text"], ["Oui"]):
            selected = detect_checkbox_or_marker_selected(
                near_heading, w, ["Oui"], img_bgr=img_bgr, x_gap_max=70, y_pad=18
            )
            if selected is True:
                result["control_overall"] = "Oui"
                break

    if result["control_overall"] is None:
        for w in near_heading:
            if text_matches(w["text"], ["Non"]):
                selected = detect_checkbox_or_marker_selected(
                    near_heading, w, ["Non"], img_bgr=img_bgr, x_gap_max=70, y_pad=18
                )
                if selected is True:
                    result["control_overall"] = "Non"
                    break

    temp_anchor = find_best_word(words, ["Température", "Temperature", "Fait Température"])
    paras_anchor = find_best_word(words, ["Parasitologie"])
    dens_anchor = find_best_word(words, ["Densité parasitaire", "Densite parasitaire"])

    temp_x_min = temp_anchor["x1"] - 25 if temp_anchor else None
    temp_x_max = temp_anchor["x2"] + 120 if temp_anchor else None

    paras_x_min = paras_anchor["x1"] - 260 if paras_anchor else None
    paras_x_max = paras_anchor["x2"] + 280 if paras_anchor else None

    dens_x_min = dens_anchor["x1"] - 70 if dens_anchor else None
    dens_x_max = dens_anchor["x2"] + 200 if dens_anchor else None

    for idx, (row_label, row_anchor) in enumerate(rows_found):
        if idx == 0:
            row_top = row_anchor["cy"] - 18
        else:
            prev_anchor = rows_found[idx - 1][1]
            row_top = (prev_anchor["cy"] + row_anchor["cy"]) / 2.0

        if idx + 1 < len(rows_found):
            next_anchor = rows_found[idx + 1][1]
            row_bottom = (row_anchor["cy"] + next_anchor["cy"]) / 2.0
            next_row_cy = next_anchor["cy"]
        else:
            row_bottom = row_anchor["cy"] + 22
            next_row_cy = None

        row_words = window_words(words, y1=row_top, y2=row_bottom)

        row = {
            "row": row_label,
            "fait": None,
            "temperature": None,
            "parasitologie": [],
            "densite_parasitaire": []
        }

        row_near_words = [w for w in row_words if abs(w["cy"] - row_anchor["cy"]) <= 20]

        row_oui_word = None
        row_non_word = None

        for w in row_near_words:
            if text_matches(w["text"], ["Oui"]):
                row_oui_word = w
            elif text_matches(w["text"], ["Non"]):
                row_non_word = w

        explicit_fait = None
        if row_oui_word:
            sel = detect_checkbox_or_marker_selected(
                row_near_words, row_oui_word, ["Oui"], img_bgr=img_bgr, x_gap_max=105, y_pad=20
            )
            if sel is True:
                explicit_fait = "Oui"

        if explicit_fait is None and row_non_word:
            sel = detect_checkbox_or_marker_selected(
                row_near_words, row_non_word, ["Non"], img_bgr=img_bgr, x_gap_max=105, y_pad=20
            )
            if sel is True:
                explicit_fait = "Non"

        if temp_x_min is not None and temp_x_max is not None:
            temp_words = window_words(row_words, x1=temp_x_min, x2=temp_x_max)
            row["temperature"] = closest_numeric_temperature(
                temp_words,
                row_anchor,
                next_row_cy=next_row_cy,
                x_min=temp_x_min,
                x_max=temp_x_max,
                max_row_dist=24,
            )

        paras_words = row_words
        if paras_x_min is not None and paras_x_max is not None:
            paras_words = window_words(row_words, x1=paras_x_min, x2=paras_x_max)

        paras_words = [w for w in paras_words if abs(w["cy"] - row_anchor["cy"]) <= 24]

        for canonical, variants in paras_specs:
            for w in paras_words:
                if text_matches(w["text"], variants):
                    if canonical not in row["parasitologie"]:
                        row["parasitologie"].append(canonical)
                    break

        dens_words = row_words
        if dens_x_min is not None and dens_x_max is not None:
            dens_words = window_words(row_words, x1=dens_x_min, x2=dens_x_max)

        dens_words = [w for w in dens_words if abs(w["cy"] - row_anchor["cy"]) <= 18]

        for canonical, variants in dens_specs:
            opts = [w for w in dens_words if text_matches(w["text"], variants)]
            for ow in opts:
                sel = detect_checkbox_or_marker_selected(
                    dens_words, ow, variants, img_bgr=img_bgr, x_gap_max=100, y_pad=16
                )
                if sel is True and canonical not in row["densite_parasitaire"]:
                    row["densite_parasitaire"].append(canonical)

        if explicit_fait is not None:
            row["fait"] = explicit_fait
        elif row["temperature"] or row["parasitologie"] or row["densite_parasitaire"]:
            row["fait"] = "Oui"
        else:
            row["fait"] = None

        result["rows"].append(row)

    commentaires_anchor = find_word_near(
        words,
        ["Commentaires & Remarques", "Commentaires", "Remarques"],
        x1=control_anchor["x1"] - 40,
        x2=control_anchor["x1"] + 900,
        y1=control_anchor["y1"] + 180,
        y2=control_anchor["y1"] + 420,
    )

    perdu_anchor = find_word_near(
        words,
        ["Perdu de vue"],
        x1=control_anchor["x1"] - 80,
        x2=control_anchor["x1"] + 950,
        y1=control_anchor["y1"] + 180,
        y2=control_anchor["y1"] + 480,
    )

    if commentaires_anchor:
        comment_words = window_words(
            words,
            x1=commentaires_anchor["x1"] - 20,
            x2=commentaires_anchor["x1"] + 850,
            y1=commentaires_anchor["y2"] - 5,
            y2=(perdu_anchor["y1"] - 5) if perdu_anchor else (commentaires_anchor["y2"] + 90),
        )
        comment_texts = []
        for w in sorted(comment_words, key=lambda z: (z["y1"], z["x1"])):
            txt = clean_text(w["text"])
            if not txt:
                continue
            if text_matches(txt, ["Perdu de vue"]):
                continue
            if txt not in comment_texts:
                comment_texts.append(txt)

        if comment_texts:
            result["commentaires_remarques"] = " ".join(comment_texts)

    if perdu_anchor:
        lost_zone = window_words(
            words,
            x1=perdu_anchor["x1"] - 160,
            x2=perdu_anchor["x2"] + 60,
            y1=perdu_anchor["y1"] - 30,
            y2=perdu_anchor["y2"] + 30,
        )

        sel = detect_checkbox_or_marker_selected(
            lost_zone,
            perdu_anchor,
            ["Perdu de vue"],
            img_bgr=img_bgr,
            x_gap_max=150,
            y_pad=26,
        )
        if sel is True:
            result["perdu_de_vue"] = "Oui"
        elif sel is False:
            result["perdu_de_vue"] = "Non"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse contrôle parasitologique block from OCR JSON with optional image-based checkbox detection"
    )
    parser.add_argument("ocr_json_path", help="Path to OCR JSON with word boxes")
    parser.add_argument("--page-image-path", default=None, help="Optional page image for checkbox detection")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page4_controle_parasito_visual",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(ocr_json_path)
    words = extract_page_words(data, page_num=args.page_num)

    img_bgr = None
    if args.page_image_path:
        img_bgr = cv2.imread(str(args.page_image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {args.page_image_path}")

    result = parse_controle_block(words, img_bgr=img_bgr)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": args.page_image_path,
        "page_num": args.page_num,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_controle_parasito_visual.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded:       {len(words)}")
    print(f"Saved JSON:         {out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
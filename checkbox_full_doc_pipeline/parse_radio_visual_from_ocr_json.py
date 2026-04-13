from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np

from debug_visual_utils import ensure_dir, draw_many_boxes, save_image, crop_box


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

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


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def overlap_ratio(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    denom = min(box_area(box_a), box_area(box_b))
    if denom <= 0:
        return 0.0
    return inter / denom


# --------------------------------------------------
# OCR JSON loading
# --------------------------------------------------

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
        })

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

    if isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            txt = w.get("text") or w.get("word_text") or w.get("transcription")
            box = w.get("box") or w.get("bbox") or w.get("points")
            add_word(txt, box)
        return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    pages = data.get("pages") or data.get("results") or data.get("page_results")
    if isinstance(pages, list):
        idx = page_num - 1
        if 0 <= idx < len(pages):
            page = pages[idx]
            if isinstance(page, dict):
                if "rec_texts" in page and "dt_polys" in page:
                    for txt, box in zip(page["rec_texts"], page["dt_polys"]):
                        add_word(txt, box)
                    return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    return sorted(out, key=lambda w: (w["y1"], w["x1"]))


# --------------------------------------------------
# Matching helpers
# --------------------------------------------------

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


def find_anchor(words, variants):
    for w in words:
        if text_matches(w["text"], variants):
            return w
    return None


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


# --------------------------------------------------
# OCR text-state helpers
# --------------------------------------------------

def parse_embedded_marker_text(word_text: str):
    raw = clean_text(word_text)
    if not raw:
        return None, raw

    if len(raw) >= 2 and raw[0] in {"O", "o", "0"}:
        return "O", clean_text(raw[1:])

    return None, raw


def option_word_state_from_text(word_text: str, label_variants: list[str]):
    marker, rest = parse_embedded_marker_text(word_text)

    if marker == "O":
        if text_matches(rest, label_variants):
            return "unselected_text"

    if text_matches(word_text, label_variants):
        return "plain_text"

    return "not_match"


# --------------------------------------------------
# Marker crop scoring
# --------------------------------------------------

def marker_crop_box_for_word(word, left_width=34, right_gap=3, y_pad=6):
    return [
        int(word["x1"] - left_width),
        int(word["y1"] - y_pad),
        int(word["x1"] - right_gap),
        int(word["y2"] + y_pad),
    ]


def score_marker_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "dark_ratio": 0.0,
            "component_count": 0,
            "largest_area": 0,
            "state": "unknown",
        }

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    h, w = th.shape[:2]
    total = float(h * w) if h > 0 and w > 0 else 1.0
    dark_ratio = float(np.count_nonzero(th)) / total

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 3:
            areas.append(area)

    component_count = len(areas)
    largest_area = max(areas) if areas else 0

    if dark_ratio >= 0.16 or largest_area >= 55:
        state = "selected_like"
    elif dark_ratio >= 0.03 or largest_area >= 10:
        state = "unselected_like"
    else:
        state = "unknown"

    return {
        "dark_ratio": round(dark_ratio, 4),
        "component_count": component_count,
        "largest_area": largest_area,
        "state": state,
        "binary": th,
    }


# --------------------------------------------------
# Field presets
# --------------------------------------------------

FIELD_PRESETS = {
    "chimioprophylaxie_oui_non_nsp": {
        "anchor_variants": ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 650, "y_before": 5, "y_after": 80},
    },
    "sexe": {
        "anchor_variants": ["Sexe"],
        "options": [
            {"label": "M", "variants": ["M"]},
            {"label": "F", "variants": ["F"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 500, "y_before": 5, "y_after": 60},
    },
    "patient_adresse": {
        "anchor_variants": ["Patient adressé par un", "Patient adresse par un", "médecin"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 500, "y_before": 5, "y_after": 55},
    },
    "consultation_avant": {
        "anchor_variants": [
            "consultation médicale avant",
            "consultation medicale avant",
            "présente consultation",
            "presente consultation",
        ],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 780, "y_before": 5, "y_after": 75},
    },
    "residence_zone_endemie": {
        "anchor_variants": [
            "Résidence durant le séjour en zone d'endémie",
            "Residence durant le sejour en zone d'endemie",
        ],
        "options": [
            {"label": "Urbain strict", "variants": ["Urbain strict"]},
            {"label": "Rural", "variants": ["Rural"]},
            {"label": "Itinérant / Mixte", "variants": ["Itinérant/Mixte", "Itinérant / Mixte", "Itinerant / Mixte"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 900, "y_before": 5, "y_after": 85},
    },
    "etat_clinique": {
        "anchor_variants": [
            "Etat clinique au moment du diagnostic",
            "État clinique au moment du diagnostic",
        ],
        "options": [
            {"label": "Accès simple sans vomissements", "variants": ["Acces simple sans vomissements", "Accès simple sans vomissements"]},
            {"label": "Accès simple AVEC vomissements", "variants": ["Acces simple AVEC vomissements", "Accès simple AVEC vomissements"]},
            {"label": "Formes Asymptomatiques et découvertes fortuites", "variants": ["Formes Asymptomatiques et découvertes fortuites"]},
            {"label": "Accès GRAVE", "variants": ["Acces GRAVE", "Accès GRAVE"]},
            {"label": "Paludisme Viscéral évolutif (PVE)", "variants": ["Paludisme Visceral evolutif", "Paludisme Viscéral évolutif", "PVE"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1000, "y_before": 5, "y_after": 130},
    },
    "evolution_clinique": {
        "anchor_variants": [
            "Evolution clinique",
            "Évolution clinique",
        ],
        "options": [
            {"label": "Guérison", "variants": ["Guerison", "Guérison"]},
            {"label": "DECES", "variants": ["DECES", "Décès", "Deces"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 700, "y_before": 5, "y_after": 75},
    },
}


# --------------------------------------------------
# Field parser
# --------------------------------------------------

def parse_radio_field(words, img_bgr, field_key, debug_dir: Path, page_stem: str):
    preset = FIELD_PRESETS[field_key]

    anchor = find_anchor(words, preset["anchor_variants"])
    result = {
        "field_key": field_key,
        "found_anchor": bool(anchor),
        "anchor_box": None,
        "selected_option": None,
        "options": [],
    }

    if not anchor:
        return result

    anchor_box = [anchor["x1"], anchor["y1"], anchor["x2"], anchor["y2"]]
    result["anchor_box"] = anchor_box

    sw = preset["search_window"]
    local_words = window_words(
        words,
        x1=anchor["x1"] - sw["x_pad_left"],
        x2=anchor["x1"] + sw["x_pad_right"],
        y1=anchor["y1"] - sw["y_before"],
        y2=anchor["y2"] + sw["y_after"],
    )

    debug_items = [
        {"box": anchor_box, "color": (255, 0, 0), "label": "anchor", "thickness": 2}
    ]

    found_count = 0

    for opt in preset["options"]:
        candidate_words = []
        for w in local_words:
            word_box = [w["x1"], w["y1"], w["x2"], w["y2"]]

            if overlap_ratio(word_box, anchor_box) > 0.5:
                continue

            text_state = option_word_state_from_text(w["text"], opt["variants"])
            if text_state != "not_match":
                candidate_words.append((w, text_state))

        if not candidate_words:
            result["options"].append({
                "label": opt["label"],
                "found_word": False,
                "state": "missing_word",
            })
            continue

        candidate_words = sorted(candidate_words, key=lambda t: (t[0]["y1"], t[0]["x1"]))
        word, text_state = candidate_words[0]
        found_count += 1

        word_box = [word["x1"], word["y1"], word["x2"], word["y2"]]
        marker_box = marker_crop_box_for_word(word)

        crop = crop_box(img_bgr, marker_box)
        score = score_marker_crop(crop)

        combined_state = "unknown"
        if text_state == "unselected_text":
            combined_state = "unselected_text"
        elif score["state"] == "selected_like":
            combined_state = "selected_like"
        elif score["state"] == "unselected_like":
            combined_state = "unselected_like"
        elif text_state == "plain_text":
            combined_state = "plain_text"

        opt_result = {
            "label": opt["label"],
            "found_word": True,
            "word_text": word["text"],
            "word_box": word_box,
            "marker_box": marker_box,
            "text_state": text_state,
            "dark_ratio": score["dark_ratio"],
            "component_count": score["component_count"],
            "largest_area": score["largest_area"],
            "state": combined_state,
        }
        result["options"].append(opt_result)

        debug_items.append({
            "box": word_box,
            "color": (0, 255, 0),
            "label": f"text:{opt['label']}",
            "thickness": 2
        })
        debug_items.append({
            "box": marker_box,
            "color": (0, 165, 255),
            "label": f"marker:{opt['label']}",
            "thickness": 2
        })

        if crop is not None:
            crop_path = debug_dir / f"{page_stem}_{field_key}_{opt['label']}_marker_crop.png"
            save_image(crop_path, crop)
            if "binary" in score:
                bin_path = debug_dir / f"{page_stem}_{field_key}_{opt['label']}_marker_binary.png"
                save_image(bin_path, score["binary"])

    plain = [o for o in result["options"] if o.get("text_state") == "plain_text"]
    explicit_o = [o for o in result["options"] if o.get("text_state") == "unselected_text"]
    strong_selected = [o for o in result["options"] if o.get("state") == "selected_like"]

    if field_key == "evolution_clinique" and len(plain) >= 1:
        plain_sorted = sorted(plain, key=lambda o: (o["word_box"][1], o["word_box"][0]))
        result["selected_option"] = plain_sorted[0]["label"]

    elif len(strong_selected) == 1:
        result["selected_option"] = strong_selected[0]["label"]

    elif len(plain) == 1 and len(explicit_o) >= 1 and (len(plain) + len(explicit_o) == found_count):
        result["selected_option"] = plain[0]["label"]

    elif field_key == "etat_clinique" and len(plain) >= 1 and len(explicit_o) >= 1:
        plain_sorted = sorted(plain, key=lambda o: (o["word_box"][1], o["word_box"][0]))
        result["selected_option"] = plain_sorted[0]["label"]

    dbg = draw_many_boxes(img_bgr, debug_items)
    save_image(debug_dir / f"{page_stem}_{field_key}_debug.png", dbg)

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse radio fields visually from OCR JSON + page image")
    parser.add_argument("ocr_json_path", help="Path to OCR JSON")
    parser.add_argument("page_image_path", help="Path to rendered page image")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--field-key",
        required=True,
        choices=sorted(FIELD_PRESETS.keys()),
        help="Which predefined field to parse",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_parser",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    page_image_path = Path(args.page_image_path)
    out_dir = Path(args.output_dir)
    debug_dir = out_dir / "debug"
    ensure_dir(out_dir)
    ensure_dir(debug_dir)

    data = load_json(ocr_json_path)
    data = data.get("res", data)
    words = extract_page_words(data, page_num=args.page_num)

    img_bgr = cv2.imread(str(page_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {page_image_path}")

    page_stem = f"{ocr_json_path.stem}_page{args.page_num}"
    result = parse_radio_field(words, img_bgr, args.field_key, debug_dir, page_stem)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(page_image_path),
        "page_num": args.page_num,
        "field_key": args.field_key,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_{args.field_key}_visual.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded: {len(words)}")
    print(f"Saved JSON:   {out_json}")
    print(f"Debug dir:    {debug_dir}")
    print(f"Selected:     {result.get('selected_option')}")
    print("Field result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
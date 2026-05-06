from pathlib import Path
import argparse
import json
import re


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
# OCR JSON loading
# --------------------------------------------------

def extract_page_words(data: dict, page_num: int = 1):
    out = []

    def add_word(text, box):
        if not text or not box:
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

                recs = (
                    page.get("words")
                    or page.get("ocr_results")
                    or page.get("dt_polys")
                    or page.get("rec_texts")
                )
                if isinstance(recs, list):
                    for item in recs:
                        if isinstance(item, dict):
                            txt = item.get("text") or item.get("word_text") or item.get("transcription")
                            box = item.get("box") or item.get("bbox") or item.get("points") or item.get("poly")
                            add_word(txt, box)
                    return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("word_text") or item.get("transcription")
                box = item.get("box") or item.get("bbox") or item.get("points") or item.get("poly")
                add_word(txt, box)

    return sorted(out, key=lambda w: (w["y1"], w["x1"]))


# --------------------------------------------------
# Search helpers
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


# --------------------------------------------------
# Marker helpers
# --------------------------------------------------

def token_is_selected_marker_only(w) -> bool:
    s = clean_text(w["text"])
    return s in {"X", "x", "×"}


def token_is_unselected_marker_only(w) -> bool:
    s = clean_text(w["text"])
    return s in {"O", "o", "□"}


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
            if prefix in {"o", "□"}:
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


def detect_option_selected(row_words, option_word, variants, x_gap_max=65, y_pad=12):
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

    return False


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


# --------------------------------------------------
# Main parser
# --------------------------------------------------

def parse_controle_block(words):
    result = {
        "field": "Contrôle parasitologique P falciparum",
        "found": False,
        "control_overall": None,
        "rows": []
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
        y2=control_anchor["y2"] + 260,
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

    # -----------------------------
    # Overall Oui / Non
    # -----------------------------
    near_heading = window_words(
        words,
        x1=control_anchor["x1"] - 10,
        x2=control_anchor["x1"] + 400,
        y1=control_anchor["y1"] - 10,
        y2=control_anchor["y2"] + 45,
    )

    for w in near_heading:
        if text_matches(w["text"], ["Oui"]):
            if detect_option_selected(near_heading, w, ["Oui"], x_gap_max=55, y_pad=12):
                result["control_overall"] = "Oui"
                break

    if result["control_overall"] is None:
        for w in near_heading:
            if text_matches(w["text"], ["Non"]):
                if detect_option_selected(near_heading, w, ["Non"], x_gap_max=55, y_pad=12):
                    result["control_overall"] = "Non"
                    break

    # -----------------------------
    # Column anchors
    # -----------------------------
    temp_anchor = find_best_word(words, ["Température", "Temperature", "Fait Température"])
    paras_anchor = find_best_word(words, ["Parasitologie"])
    dens_anchor = find_best_word(words, ["Densité parasitaire", "Densite parasitaire"])

    temp_x_min = temp_anchor["x1"] - 25 if temp_anchor else None
    temp_x_max = temp_anchor["x2"] + 120 if temp_anchor else None

    paras_x_min = paras_anchor["x1"] - 120 if paras_anchor else None
    paras_x_max = paras_anchor["x2"] + 190 if paras_anchor else None

    dens_x_min = dens_anchor["x1"] - 60 if dens_anchor else None
    dens_x_max = dens_anchor["x2"] + 180 if dens_anchor else None

    # -----------------------------
    # Parse each row
    # -----------------------------
    for idx, (row_label, row_anchor) in enumerate(rows_found):
        if idx == 0:
            row_top = row_anchor["cy"] - 14
        else:
            prev_anchor = rows_found[idx - 1][1]
            row_top = (prev_anchor["cy"] + row_anchor["cy"]) / 2.0

        if idx + 1 < len(rows_found):
            next_anchor = rows_found[idx + 1][1]
            row_bottom = (row_anchor["cy"] + next_anchor["cy"]) / 2.0
            next_row_cy = next_anchor["cy"]
        else:
            row_bottom = row_anchor["cy"] + 14
            next_row_cy = None

        row_words = window_words(words, y1=row_top, y2=row_bottom)

        row = {
            "row": row_label,
            "fait": None,
            "temperature": None,
            "parasitologie": [],
            "densite_parasitaire": []
        }

        # -----------------------------
        # Temperature
        # -----------------------------
        if temp_x_min is not None and temp_x_max is not None:
            temp_words = window_words(row_words, x1=temp_x_min, x2=temp_x_max)
            row["temperature"] = closest_numeric_temperature(
                temp_words,
                row_anchor,
                next_row_cy=next_row_cy,
                x_min=temp_x_min,
                x_max=temp_x_max,
                max_row_dist=18,
            )

        # -----------------------------
        # Parasitologie
        # -----------------------------
        paras_words = row_words
        if paras_x_min is not None and paras_x_max is not None:
            paras_words = window_words(row_words, x1=paras_x_min, x2=paras_x_max)

        # extra strict vertical filtering around the row center
        paras_words = [
            w for w in paras_words
            if abs(w["cy"] - row_anchor["cy"]) <= 12
        ]

        for canonical, variants in paras_specs:
            opts = [w for w in paras_words if text_matches(w["text"], variants)]
            for ow in opts:
                if detect_option_selected(paras_words, ow, variants, x_gap_max=100, y_pad=12):
                    if canonical not in row["parasitologie"]:
                        row["parasitologie"].append(canonical)

        # -----------------------------
        # Densité parasitaire
        # -----------------------------
        dens_words = row_words
        if dens_x_min is not None and dens_x_max is not None:
            dens_words = window_words(row_words, x1=dens_x_min, x2=dens_x_max)

        dens_words = [
            w for w in dens_words
            if abs(w["cy"] - row_anchor["cy"]) <= 12
        ]

        for canonical, variants in dens_specs:
            opts = [w for w in dens_words if text_matches(w["text"], variants)]
            for ow in opts:
                if detect_option_selected(dens_words, ow, variants, x_gap_max=80, y_pad=12):
                    if canonical not in row["densite_parasitaire"]:
                        row["densite_parasitaire"].append(canonical)

        if row["temperature"] or row["parasitologie"] or row["densite_parasitaire"]:
            row["fait"] = "Oui"
        else:
            row["fait"] = "Non"

        result["rows"].append(row)

    return result

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse page 4 contrôle parasitologique block from OCR JSON with boxes"
    )
    parser.add_argument("ocr_json_path", help="Path to OCR JSON with word boxes")
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
    result = parse_controle_block(words)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_num": args.page_num,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_controle_parasito_visual.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded:       {len(words)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
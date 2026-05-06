from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np

from debug_visual_utils import ensure_dir, draw_many_boxes, save_image, crop_box


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

    if isinstance(data, dict) and "overall_ocr_res" in data:
        ocr = data["overall_ocr_res"]
        if isinstance(ocr, dict) and "rec_texts" in ocr and "dt_polys" in ocr:
            for txt, box in zip(ocr["rec_texts"], ocr["dt_polys"]):
                add_word(txt, box)
            return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    return sorted(out, key=lambda w: (w["y1"], w["x1"]))


def parse_embedded_marker_text(word_text: str):
    raw = clean_text(word_text)
    if not raw:
        return None, raw
    if len(raw) >= 2 and raw[0] in {"O", "o", "0"}:
        return "O", clean_text(raw[1:])
    return None, raw


def classify_word_text_state(word_text: str):
    raw = clean_text(word_text)
    if not raw:
        return "ignore", ""

    raw_norm = norm(raw)

    if raw in {"•", "?", "·", "●", "■", "□", "日", "国", ".", "-", "三"}:
        return "marker_only", raw

    marker, rest = parse_embedded_marker_text(raw)
    if marker == "O" and clean_text(rest):
        return "unselected_text", clean_text(rest)

    bad_substrings = [
        "annuler", "precedent", "accueil", "deconnecter", "voozanoo",
        "http", "page ", "jj/mm", "aaaa", "id patient", "16/02/2007",
    ]
    if any(x in raw_norm for x in bad_substrings):
        return "ignore", raw

    return "plain_text", raw


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

    largest_area = max(areas) if areas else 0
    component_count = len(areas)

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


def group_words_into_lines(words, y_tol=14):
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["cy"], w["x1"]))
    lines = []

    for w in words_sorted:
        placed = False
        for line in lines:
            if abs(w["cy"] - line["cy"]) <= y_tol:
                line["words"].append(w)
                xs = [x["x1"] for x in line["words"]]
                ys = [x["y1"] for x in line["words"]]
                xe = [x["x2"] for x in line["words"]]
                ye = [x["y2"] for x in line["words"]]
                line["x1"] = min(xs)
                line["y1"] = min(ys)
                line["x2"] = max(xe)
                line["y2"] = max(ye)
                line["cy"] = sum(x["cy"] for x in line["words"]) / len(line["words"])
                placed = True
                break

        if not placed:
            lines.append({
                "words": [w],
                "x1": w["x1"],
                "y1": w["y1"],
                "x2": w["x2"],
                "y2": w["y2"],
                "cy": w["cy"],
            })

    for line in lines:
        line["words"] = sorted(line["words"], key=lambda x: x["x1"])
        line["text"] = clean_text(" ".join(x["text"] for x in line["words"]))
        line["box"] = [line["x1"], line["y1"], line["x2"], line["y2"]]

    return sorted(lines, key=lambda l: (l["y1"], l["x1"]))


def is_likely_heading(line_text: str) -> bool:
    t = clean_text(line_text)
    tn = norm(t)
    if not t:
        return False
    if ":" in t:
        return True
    heading_phrases = [
        "sexe", "ethnicite", "nature du sejour", "residence durant",
        "etat clinique", "patient adresse", "consultation medicale",
        "chimioprophylaxie", "protection personnelle", "bandelettes",
        "lame transmise", "evolution clinique", "autres pays d'endemie",
    ]
    return any(h in tn for h in heading_phrases)


def line_to_heading_and_inline_chunks(line):
    words = sorted(line["words"], key=lambda w: w["x1"])
    colon_idx = None

    for i, w in enumerate(words):
        if ":" in w["text"] or w["text"].endswith(":"):
            colon_idx = i
            break

    if colon_idx is None:
        txt = line["text"]
        if ":" in txt:
            left, right = txt.split(":", 1)
            return {
                "heading_text": clean_text(left + ":"),
                "inline_words": [],
                "has_inline_text": bool(clean_text(right)),
            }
        return {
            "heading_text": line["text"],
            "inline_words": [],
            "has_inline_text": False,
        }

    heading_words = words[:colon_idx + 1]
    inline_words = words[colon_idx + 1:]

    heading_text = clean_text(" ".join(w["text"] for w in heading_words))
    if not heading_text.endswith(":"):
        heading_text = heading_text + ":"

    return {
        "heading_text": heading_text,
        "inline_words": inline_words,
        "has_inline_text": bool(inline_words),
    }


def split_words_into_option_chunks(words, gap_threshold=55):
    words = sorted(words, key=lambda w: w["x1"])
    if not words:
        return []

    chunks = []
    current = [words[0]]

    for prev, cur in zip(words, words[1:]):
        prev_state, _ = classify_word_text_state(prev["text"])
        cur_state, _ = classify_word_text_state(cur["text"])
        gap = cur["x1"] - prev["x2"]

        split_here = False

        if gap >= gap_threshold:
            split_here = True

        if cur_state in {"unselected_text", "marker_only"}:
            split_here = True

        if prev_state == "marker_only":
            split_here = False

        if split_here:
            chunks.append(current)
            current = [cur]
        else:
            current.append(cur)

    if current:
        chunks.append(current)

    out = []
    for chunk in chunks:
        x1 = min(w["x1"] for w in chunk)
        y1 = min(w["y1"] for w in chunk)
        x2 = max(w["x2"] for w in chunk)
        y2 = max(w["y2"] for w in chunk)
        out.append({
            "words": chunk,
            "text": clean_text(" ".join(w["text"] for w in chunk)),
            "box": [x1, y1, x2, y2],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return out


def split_line_into_option_chunks(line, gap_threshold=55):
    return split_words_into_option_chunks(line["words"], gap_threshold=gap_threshold)


def is_likely_option_chunk(chunk_text: str) -> bool:
    state, normalized = classify_word_text_state(chunk_text)
    tn = norm(normalized)

    if state == "ignore":
        return False
    if state in {"unselected_text", "marker_only"}:
        return True
    if state == "plain_text" and 1 <= len(tn) <= 80:
        return True
    return False


def classify_chunk_option(chunk, img_bgr):
    text = chunk["text"]
    text_state, normalized = classify_word_text_state(text)

    marker_box = marker_crop_box_for_word({
        "x1": chunk["x1"],
        "y1": chunk["y1"],
        "x2": chunk["x2"],
        "y2": chunk["y2"],
    })
    crop = crop_box(img_bgr, marker_box)
    score = score_marker_crop(crop)

    final_state = "unknown"
    if text_state == "unselected_text":
        final_state = "unselected"
    elif score["state"] == "selected_like":
        final_state = "selected"
    elif score["state"] == "unselected_like":
        final_state = "unselected"
    elif text_state == "plain_text":
        final_state = "plain"

    return {
        "raw_text": text,
        "normalized_text": normalized,
        "chunk_box": chunk["box"],
        "marker_box": marker_box,
        "text_state": text_state,
        "dark_ratio": score["dark_ratio"],
        "component_count": score["component_count"],
        "largest_area": score["largest_area"],
        "visual_state": score["state"],
        "state": final_state,
        "binary": score.get("binary"),
        "crop": crop,
    }


def looks_like_new_question(line_text: str) -> bool:
    tn = norm(line_text)
    starters = [
        "autres pays d'endemie",
        "nature du sejour",
        "residence durant",
        "frequence des sejours",
        "etat clinique",
        "patient adresse",
        "consultation medicale",
        "chimioprophylaxie",
        "protection personnelle",
        "bandelettes",
        "autres techniques",
        "lame transmise",
        "evolution clinique",
    ]
    return any(s in tn for s in starters) or ":" in line_text


def is_valid_option_group(group_heading: str, options: list[dict]) -> bool:
    if not options:
        return False

    heading_norm = norm(group_heading)

    # obvious non-option text fields
    reject_heading_substrings = [
        "annee", "prenom", "nom", "date de naissance", "date depart",
        "date retour", "pays de residence", "paysde naissance",
        "si autres,preciser", "si autres, préciser", "si militaire",
        "id", "localisation",
    ]
    if any(x in heading_norm for x in reject_heading_substrings):
        return False

    meaningful = [o for o in options if o["state"] in {"selected", "unselected", "plain"}]
    if len(meaningful) < 2:
        return False

    unselected = [o for o in meaningful if o["state"] == "unselected"]
    selected = [o for o in meaningful if o["state"] == "selected"]
    plain = [o for o in meaningful if o["state"] == "plain"]

    # strongest evidence: at least two explicit options with one O-prefixed or one selected marker
    if len(unselected) >= 2:
        return True
    if len(selected) >= 1 and len(unselected) >= 1:
        return True

    # allow known form question headings with multiple short options
    allow_heading_substrings = [
        "sexe", "ethnicite", "nature du sejour", "residence durant",
        "etat clinique", "patient adresse", "consultation medicale",
        "chimioprophylaxie", "protection personnelle", "bandelettes",
        "autres techniques", "lame transmise", "evolution clinique",
        "duree du sejour", "autres pays d'endemie",
    ]
    if any(x in heading_norm for x in allow_heading_substrings) and len(meaningful) >= 2:
        return True

    # otherwise reject
    return False


def extract_option_groups(lines, img_bgr, debug_dir: Path, page_stem: str):
    groups = []
    current_heading = None
    current_options = []
    debug_items = []

    def flush_group():
        nonlocal current_heading, current_options, groups
        if current_heading and current_options:
            if not is_valid_option_group(current_heading["text"], current_options):
                return

            found = [o for o in current_options if o["state"] in {"selected", "unselected", "plain"}]
            selected = [o for o in found if o["state"] == "selected"]
            unselected = [o for o in found if o["state"] == "unselected"]
            plain = [o for o in found if o["state"] == "plain"]

            inferred_selected = None
            if len(selected) == 1:
                inferred_selected = selected[0]["normalized_text"]
            elif len(selected) == 0 and len(plain) == 1 and len(unselected) >= 1:
                inferred_selected = plain[0]["normalized_text"]

            groups.append({
                "heading": current_heading["text"],
                "heading_box": current_heading["box"],
                "inferred_selected": inferred_selected,
                "options": [{k: v for k, v in o.items() if k not in {"binary", "crop"}} for o in current_options],
            })

    for line in lines:
        text = line["text"]

        if is_likely_heading(text):
            flush_group()

            split_info = line_to_heading_and_inline_chunks(line)
            heading_text = split_info["heading_text"]

            current_heading = {
                "text": heading_text,
                "box": line["box"],
                "words": line["words"],
                "y1": line["y1"],
                "y2": line["y2"],
            }
            current_options = []

            debug_items.append({
                "box": line["box"],
                "color": (255, 0, 0),
                "label": f"heading:{heading_text[:28]}",
                "thickness": 2,
            })

            if split_info["inline_words"]:
                inline_chunks = split_words_into_option_chunks(split_info["inline_words"], gap_threshold=45)
                for chunk in inline_chunks:
                    if not is_likely_option_chunk(chunk["text"]):
                        continue
                    opt = classify_chunk_option(chunk, img_bgr)
                    current_options.append(opt)

                    label = f"{opt['state']}:{opt['normalized_text'][:18]}"
                    color = {
                        "selected": (0, 255, 0),
                        "unselected": (0, 165, 255),
                        "plain": (255, 255, 0),
                        "unknown": (128, 128, 128),
                    }.get(opt["state"], (128, 128, 128))

                    debug_items.append({
                        "box": opt["chunk_box"],
                        "color": color,
                        "label": label,
                        "thickness": 2,
                    })
                    debug_items.append({
                        "box": opt["marker_box"],
                        "color": (255, 0, 255),
                        "label": "marker",
                        "thickness": 1,
                    })

            continue

        if current_heading is not None:
            vertical_gap = line["y1"] - current_heading["y2"]
            if vertical_gap > 220 or looks_like_new_question(text):
                flush_group()
                current_heading = None
                current_options = []

                if is_likely_heading(text):
                    split_info = line_to_heading_and_inline_chunks(line)
                    heading_text = split_info["heading_text"]
                    current_heading = {
                        "text": heading_text,
                        "box": line["box"],
                        "words": line["words"],
                        "y1": line["y1"],
                        "y2": line["y2"],
                    }
                    current_options = []
                else:
                    continue

            chunks = split_line_into_option_chunks(line)

            for chunk in chunks:
                if not is_likely_option_chunk(chunk["text"]):
                    continue

                opt = classify_chunk_option(chunk, img_bgr)
                current_options.append(opt)

                label = f"{opt['state']}:{opt['normalized_text'][:18]}"
                color = {
                    "selected": (0, 255, 0),
                    "unselected": (0, 165, 255),
                    "plain": (255, 255, 0),
                    "unknown": (128, 128, 128),
                }.get(opt["state"], (128, 128, 128))

                debug_items.append({
                    "box": opt["chunk_box"],
                    "color": color,
                    "label": label,
                    "thickness": 2,
                })
                debug_items.append({
                    "box": opt["marker_box"],
                    "color": (255, 0, 255),
                    "label": "marker",
                    "thickness": 1,
                })

    flush_group()

    dbg = draw_many_boxes(img_bgr, debug_items)
    save_image(debug_dir / f"{page_stem}_generic_groups_debug.png", dbg)

    return groups


def main():
    parser = argparse.ArgumentParser(description="Generic visual option-group extractor from OCR JSON + page image")
    parser.add_argument("ocr_json_path")
    parser.add_argument("page_image_path")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_generic",
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

    lines = group_words_into_lines(words, y_tol=14)
    page_stem = f"{ocr_json_path.stem}_page{args.page_num}"
    groups = extract_option_groups(lines, img_bgr, debug_dir, page_stem)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(page_image_path),
        "page_num": args.page_num,
        "word_count": len(words),
        "line_count": len(lines),
        "groups": groups,
    }

    out_json = out_dir / f"{page_stem}_generic_groups.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded: {len(words)}")
    print(f"Lines built:  {len(lines)}")
    print(f"Groups found: {len(groups)}")
    print(f"Saved JSON:   {out_json}")
    print(f"Debug dir:    {debug_dir}")

    for i, g in enumerate(groups, 1):
        print(f"\n[{i}] Heading: {g['heading']}")
        print(f"    Inferred selected: {g['inferred_selected']}")
        for opt in g["options"]:
            print(f"    - {opt['state']}: {opt['normalized_text']}")


if __name__ == "__main__":
    main()
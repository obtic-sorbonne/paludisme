from pathlib import Path
import argparse
import json
import re


def norm(s: str) -> str:
    return " ".join(
        s.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ï", "i")
        .replace("’", "'")
        .split()
    )


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(d: dict):
    return d.get("res", d)


def get_pp_table_bbox(pp_data: dict):
    root = get_root(pp_data)
    blocks = root.get("parsing_res_list", [])
    table_blocks = [b for b in blocks if b.get("block_label") == "table"]
    if not table_blocks:
        return None
    best = max(
        table_blocks,
        key=lambda b: (b["block_bbox"][2] - b["block_bbox"][0]) * (b["block_bbox"][3] - b["block_bbox"][1]),
    )
    return best["block_bbox"]


def get_paddle_ocr_lines(paddle_data: dict):
    root = get_root(paddle_data)
    ocr = root.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts", [])
    boxes = ocr.get("rec_boxes", [])

    lines = []
    for t, b in zip(texts, boxes):
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        lines.append({
            "text": t,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })
    return lines


def inside(box, region, pad=0):
    return (
        box["x1"] >= region[0] - pad and
        box["y1"] >= region[1] - pad and
        box["x2"] <= region[2] + pad and
        box["y2"] <= region[3] + pad
    )


def group_lines_into_rows(lines, y_thresh=18):
    if not lines:
        return []

    lines = sorted(lines, key=lambda x: (x["cy"], x["x1"]))
    rows = []
    current = [lines[0]]
    current_y = lines[0]["cy"]

    for line in lines[1:]:
        if abs(line["cy"] - current_y) <= y_thresh:
            current.append(line)
            current_y = sum(l["cy"] for l in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda x: x["x1"]))
            current = [line]
            current_y = line["cy"]

    rows.append(sorted(current, key=lambda x: x["x1"]))
    return rows


def merge_close_texts_in_row(row, x_gap=20):
    if not row:
        return []

    merged = []
    current = row[0].copy()

    for item in row[1:]:
        if item["x1"] - current["x2"] <= x_gap:
            current["text"] = f"{current['text']} {item['text']}"
            current["x2"] = max(current["x2"], item["x2"])
            current["y1"] = min(current["y1"], item["y1"])
            current["y2"] = max(current["y2"], item["y2"])
            current["cx"] = (current["x1"] + current["x2"]) / 2
            current["cy"] = (current["y1"] + current["y2"]) / 2
        else:
            merged.append(current)
            current = item.copy()

    merged.append(current)
    return merged


def detect_column_boundaries(table_bbox):
    x1, _, x2, _ = table_bbox
    width = x2 - x1
    c1 = x1 + 0.47 * width
    c2 = x1 + 0.62 * width
    c3 = x1 + 0.80 * width
    c4 = x1 + 0.93 * width
    return [c1, c2, c3, c4]


def assign_to_columns(row, boundaries):
    cols = ["", "", "", "", ""]
    for item in row:
        x = item["cx"]
        text = item["text"].strip()

        if x < boundaries[0]:
            idx = 0
        elif x < boundaries[1]:
            idx = 1
        elif x < boundaries[2]:
            idx = 2
        elif x < boundaries[3]:
            idx = 3
        else:
            idx = 4

        cols[idx] = (cols[idx] + " " + text).strip() if cols[idx] else text
    return cols


def is_noise_row(cols):
    joined = " ".join(cols).strip()
    return not joined or joined in {"网", "\\", "……"}


def is_metadata_row(cols):
    row = norm(" ".join(cols))
    metadata_keywords = [
        "nom patient", "date / heure", "date naissance", "prescripteur",
        "adresse", "patient adresse", "copie à", "echantillon", "prelevement",
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2", "page 2 sur 2"
    ]
    return any(k in row for k in metadata_keywords)


def looks_like_header(cols):
    row = norm(" ".join(cols))
    keys = ["description", "resultat", "unite", "valeurs normales", "val."]
    return sum(1 for k in keys if k in row) >= 2


def looks_like_lab_row(cols):
    row = " ".join(cols)
    has_num = bool(re.search(r"\d+[.,]\d+|\b\d+\b", row))
    has_unit = bool(re.search(r"%|g/dl|g/100ml|pg/hematie|10x3/mm3|10X6/mm3|μ×3|ux3", row, re.I))
    has_text = bool(re.search(r"[A-Za-zÀ-ÿ]", row))
    return has_text and (has_num or has_unit)


def looks_like_note_anchor(cols):
    row = norm(" ".join(cols))
    anchors = [
        "rech parasites sang",
        "ag plasmodium",
        "notion de voyage recent",
        "pays d' origine",
    ]
    return any(a in row for a in anchors)


def main():
    parser = argparse.ArgumentParser(description="Merge PP-Structure table region with plain Paddle OCR text")
    parser.add_argument("pp_json", help="PP-Structure page JSON")
    parser.add_argument("paddle_json", help="Plain Paddle OCR page JSON")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/merged_tables",
        help="Output directory",
    )
    args = parser.parse_args()

    pp_json = Path(args.pp_json)
    paddle_json = Path(args.paddle_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pp_data = load_json(pp_json)
    paddle_data = load_json(paddle_json)

    table_bbox = get_pp_table_bbox(pp_data)
    if table_bbox is None:
        print("No table block found in PP-Structure JSON.")
        print("Top-level keys:", list(pp_data.keys())[:20])
        root = get_root(pp_data)
        print("Root keys:", list(root.keys())[:20])
        return

    ocr_lines = get_paddle_ocr_lines(paddle_data)
    table_lines = [ln for ln in ocr_lines if inside(ln, table_bbox, pad=10)]

    if not table_lines:
        print("No OCR lines found inside detected table block.")
        return

    rows = group_lines_into_rows(table_lines, y_thresh=18)
    rows = [merge_close_texts_in_row(r, x_gap=20) for r in rows]
    boundaries = detect_column_boundaries(table_bbox)

    final_rows = []
    trailing_notes = []
    note_mode = False

    for row in rows:
        cols = assign_to_columns(row, boundaries)

        if is_noise_row(cols) or is_metadata_row(cols):
            continue
        if looks_like_header(cols):
            final_rows.append(["Description", "Résultat", "Unité", "Valeurs normales", "Val."])
            continue
        if looks_like_note_anchor(cols):
            note_mode = True
            trailing_notes.append(" | ".join([c for c in cols if c]))
            continue
        if note_mode:
            joined = " | ".join([c for c in cols if c]).strip()
            if joined:
                trailing_notes.append(joined)
            continue
        if looks_like_lab_row(cols):
            final_rows.append(cols)

    out_file = out_dir / f"{pp_json.stem}_merged_table.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== STRUCTURED TABLE ROWS ===\n\n")
        for row in final_rows:
            f.write(" | ".join(row) + "\n")
        if trailing_notes:
            f.write("\n=== TRAILING NOTE / COMMENT SECTION ===\n\n")
            for line in trailing_notes:
                f.write(line + "\n")

    print(f"Table bbox: {table_bbox}")
    print(f"Saved merged output: {out_file}")
    print(f"Kept {len(final_rows)} structured row(s)")
    print(f"Kept {len(trailing_notes)} trailing note line(s)")


if __name__ == "__main__":
    main()
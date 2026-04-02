from pathlib import Path
import argparse
import json
import re


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(d: dict):
    return d.get("res", d)


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
    return [int(v) for v in best["block_bbox"]]


def get_paddle_lines(paddle_data: dict):
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


def get_tatr_bbox(tatr_txt: Path):
    text = tatr_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in text:
        if line.startswith("table\t"):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            box_txt = parts[2].strip()
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", box_txt)
            if len(nums) >= 4:
                x1, y1, x2, y2 = map(float, nums[:4])
                return [int(x1), int(y1), int(x2), int(y2)]
    return None


def get_tatr_render_size(tatr_txt: Path):
    """
    Reads image size from detections txt if present.
    Expected optional lines like:
      Render width: 2502
      Render height: 3516
    Falls back to None if missing.
    """
    text = tatr_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    w = None
    h = None
    for line in text:
        if line.lower().startswith("render width:"):
            try:
                w = int(line.split(":")[1].strip())
            except Exception:
                pass
        if line.lower().startswith("render height:"):
            try:
                h = int(line.split(":")[1].strip())
            except Exception:
                pass
    if w is not None and h is not None:
        return w, h
    return None


def rescale_box(box, src_w, src_h, dst_w, dst_h):
    sx = dst_w / src_w
    sy = dst_h / src_h
    x1, y1, x2, y2 = box
    return [
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    ]


def inside(line, box, pad=0):
    return (
        line["x1"] >= box[0] - pad and
        line["y1"] >= box[1] - pad and
        line["x2"] <= box[2] + pad and
        line["y2"] <= box[3] + pad
    )


def group_rows(lines, y_thresh=18):
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


def merge_close_in_row(row, x_gap=18):
    if not row:
        return []

    merged = []
    cur = row[0].copy()

    for item in row[1:]:
        if item["x1"] - cur["x2"] <= x_gap:
            cur["text"] = f"{cur['text']} {item['text']}"
            cur["x2"] = max(cur["x2"], item["x2"])
            cur["y1"] = min(cur["y1"], item["y1"])
            cur["y2"] = max(cur["y2"], item["y2"])
            cur["cx"] = (cur["x1"] + cur["x2"]) / 2
            cur["cy"] = (cur["y1"] + cur["y2"]) / 2
        else:
            merged.append(cur)
            cur = item.copy()

    merged.append(cur)
    return merged


def detect_col_boundaries(tatr_box):
    x1, _, x2, _ = tatr_box
    w = x2 - x1
    return [
        x1 + 0.45 * w,
        x1 + 0.62 * w,
        x1 + 0.79 * w,
        x1 + 0.94 * w,
    ]


def assign_cols(row, bounds):
    cols = ["", "", "", "", ""]
    for item in row:
        x = item["cx"]
        text = item["text"].strip()
        if x < bounds[0]:
            idx = 0
        elif x < bounds[1]:
            idx = 1
        elif x < bounds[2]:
            idx = 2
        elif x < bounds[3]:
            idx = 3
        else:
            idx = 4
        cols[idx] = (cols[idx] + " " + text).strip() if cols[idx] else text
    return cols


def is_noise_text(text):
    t = text.strip()
    return (not t) or t in {"网", "回", "国", "\\", "……", "□"}


def clean_text(text):
    text = text.strip()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def row_text(row):
    return clean_text(" | ".join([x["text"] for x in row if not is_noise_text(x["text"])]))


def looks_like_lab_row(cols):
    row = " ".join(cols)
    has_num = bool(re.search(r"\d+[.,]\d+|\b\d+\b", row))
    has_unit = bool(re.search(r"%|g/dl|g/100ml|pg/hematie|10x3/mm3|10X6/mm3|μ×3|ux3", row, re.I))
    has_text = bool(re.search(r"[A-Za-zÀ-ÿ]", row))
    return has_text and (has_num or has_unit)


def looks_like_header_text(text):
    t = norm(text)
    keys = ["description", "resultat", "unite", "valeurs normales", "val"]
    return sum(1 for k in keys if k in t) >= 2


def looks_like_metadata_text(text):
    t = norm(text)
    keys = [
        "nom patient", "date / heure", "date naissance", "prescripteur",
        "adresse", "patient adresse", "copie a", "echantillon", "prelevement",
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2", "page 2 sur 2"
    ]
    return any(k in t for k in keys)


def looks_like_note_anchor_text(text):
    t = norm(text)
    anchors = [
        "rech parasites sang",
        "ag plasmodium",
        "notion de voyage recent",
        "pays d' origine",
    ]
    return any(a in t for a in anchors)


def main():
    parser = argparse.ArgumentParser(description="Hybrid parser: PP-Structure + TATR + Paddle OCR")
    parser.add_argument("pp_json", help="PP-Structure page json")
    parser.add_argument("paddle_json", help="Plain Paddle/page OCR json")
    parser.add_argument("tatr_txt", help="TATR detections txt")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables",
        help="Output directory",
    )
    parser.add_argument("--tatr-width", type=int, default=None, help="TATR render width if not saved in txt")
    parser.add_argument("--tatr-height", type=int, default=None, help="TATR render height if not saved in txt")
    args = parser.parse_args()

    pp_data = load_json(Path(args.pp_json))
    paddle_data = load_json(Path(args.paddle_json))
    tatr_box_raw = get_tatr_bbox(Path(args.tatr_txt))
    pp_box = get_pp_table_bbox(pp_data)

    if pp_box is None:
        print("No PP-Structure table box found.")
        return
    if tatr_box_raw is None:
        print("No TATR table box found.")
        return

    pp_root = get_root(pp_data)
    pp_width = int(pp_root["width"])
    pp_height = int(pp_root["height"])

    # Try to read render size from txt, otherwise use CLI args, otherwise fallback
    tatr_size = get_tatr_render_size(Path(args.tatr_txt))
    if tatr_size is not None:
        tatr_render_w, tatr_render_h = tatr_size
    elif args.tatr_width and args.tatr_height:
        tatr_render_w, tatr_render_h = args.tatr_width, args.tatr_height
    else:
        # fallback based on your current rendered example
        tatr_render_w, tatr_render_h = 2502, 3516

    tatr_box = rescale_box(
        tatr_box_raw,
        tatr_render_w,
        tatr_render_h,
        pp_width,
        pp_height,
    )

    lines = get_paddle_lines(paddle_data)

    upper_lines = []
    core_lines = []
    lower_lines = []

    for ln in lines:
        if not inside(ln, pp_box, pad=10):
            continue

        if ln["y2"] < tatr_box[1]:
            upper_lines.append(ln)
        elif inside(ln, tatr_box, pad=10):
            core_lines.append(ln)
        elif ln["y1"] > tatr_box[3]:
            lower_lines.append(ln)

    # upper section
    upper_rows = group_rows(upper_lines, y_thresh=18)
    upper_rows = [merge_close_in_row(r, x_gap=20) for r in upper_rows]
    upper_text = []
    for row in upper_rows:
        line = row_text(row)
        if line and not looks_like_metadata_text(line):
            upper_text.append(line)

    # core table
    core_rows = group_rows(core_lines, y_thresh=18)
    core_rows = [merge_close_in_row(r, x_gap=20) for r in core_rows]
    bounds = detect_col_boundaries(tatr_box)

    structured_rows = []
    for row in core_rows:
        cols = [clean_text(c) for c in assign_cols(row, bounds)]
        joined = " ".join(cols).strip()
        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue
        structured_rows.append(cols)

    # lower section
    lower_rows = group_rows(lower_lines, y_thresh=18)
    lower_rows = [merge_close_in_row(r, x_gap=20) for r in lower_rows]
    lower_text = []
    for row in lower_rows:
        line = row_text(row)
        if line:
            lower_text.append(line)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.pp_json).stem}_hybrid.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"PP box: {pp_box}\n")
        f.write(f"TATR raw box: {tatr_box_raw}\n")
        f.write(f"TATR scaled box: {tatr_box}\n")
        f.write(f"PP page size: {pp_width} x {pp_height}\n")
        f.write(f"TATR render size: {tatr_render_w} x {tatr_render_h}\n\n")

        f.write("=== UPPER SECTION (inside PP box, above TATR box) ===\n\n")
        for line in upper_text:
            f.write(line + "\n")

        f.write("\n=== CORE TABLE (inside TATR box) ===\n\n")
        for row in structured_rows:
            f.write(" | ".join(row) + "\n")

        f.write("\n=== LOWER SECTION / NOTES (inside PP box, below TATR box) ===\n\n")
        for line in lower_text:
            f.write(line + "\n")

    print(f"Saved hybrid output: {out_file}")
    print(f"Upper lines: {len(upper_text)}")
    print(f"Core rows: {len(structured_rows)}")
    print(f"Lower lines: {len(lower_text)}")


if __name__ == "__main__":
    main()
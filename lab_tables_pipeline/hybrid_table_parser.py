from pathlib import Path
import argparse
import json
import re
from html.parser import HTMLParser


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
        .replace("î", "i")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("'", "'")
        .split()
    )


def clean_text(text):
    text = str(text).strip()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


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
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


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
    return (not t) or t in {
        "网", "回", "国", "\\", "……", "□", "■", "☑", "√", "✔", "✗", "✘", "区", "日"
    }


def row_text(row):
    return clean_text(" | ".join([x["text"] for x in row if not is_noise_text(x["text"])]))


def looks_like_header_text(text):
    t = norm(text)
    keys = ["description", "resultat", "unite", "valeurs normales", "val"]
    return sum(1 for k in keys if k in t) >= 2


def looks_like_metadata_text(text):
    t = norm(text)
    keys = [
        "nom patient", "date / heure", "date naissance", "prescripteur",
        "adresse", "patient adresse", "copie a", "echantillon", "prelevement",
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2", "page 2 sur 2",
        "hopital robert debre", "urgent"
    ]
    return any(k in t for k in keys)


def looks_like_note_anchor_text(text):
    t = norm(text)
    anchors = [
        "rech parasites sang",
        "ag plasmodium",
        "notion de voyage recent",
        "pays d' origine",
        "valide par",
        "technicien hemato",
        "cytologie",
    ]
    return any(a in t for a in anchors)


def count_nonempty(cols):
    return sum(1 for c in cols if clean_text(c))


def row_has_description_and_value(cols):
    return bool(clean_text(cols[0])) and any(clean_text(c) for c in cols[1:])


def structured_rows_quality(rows):
    if not rows:
        return 0.0

    good = 0
    for row in rows:
        nonempty = count_nonempty(row)
        if row_has_description_and_value(row):
            good += 1
        elif nonempty >= 3:
            good += 1

    return good / max(len(rows), 1)


def cleanup_bad_rows(rows):
    cleaned = []
    for row in rows:
        row = [clean_text(c) for c in row]
        if all(not c for c in row):
            continue
        joined = " ".join(row)
        if looks_like_metadata_text(joined):
            continue
        cleaned.append(row)
    return cleaned


def is_classic_hematology_page_from_texts(lines):
    """Check if this is a classic hematology page with standard analytes."""
    joined = "\n".join(clean_text(x["text"]) for x in lines if clean_text(x["text"]))
    j = norm(joined)
    analytes = [
        "erythrocytes", "hemoglobine", "hematocrite", "leucocytes",
        "plaquettes", "lymphocytes", "monocytes", "polyneutrophiles"
    ]
    hits = sum(1 for a in analytes if a in j)
    return hits >= 5


def is_hematology_ocr_incomplete(rows):
    """Check if hematology table has missing OCR data."""
    joined = "\n".join(" | ".join(r) for r in rows)
    j = norm(joined)

    top_hits = sum(1 for a in [
        "erythrocytes", "hemoglobine", "hematocrite", "leucocytes", "plaquettes"
    ] if a in j)

    lower_hits = sum(1 for a in [
        "lymphocytes", "monocytes", "myelocytes", "metamyelocytes",
        "blastes", "plasmocytes", "autres cellules", "poly basophiles"
    ] if a in j)

    return top_hits >= 4 and lower_hits <= 2


class SimpleHTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.in_td = False
        self.current_row = []
        self.current_cell = []
        self.current_table = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_td = True
            self.current_cell = []

    def handle_data(self, data):
        if self.in_td:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th"):
            self.in_td = False
            cell = clean_text("".join(self.current_cell))
            self.current_row.append(cell)
        elif tag == "tr":
            if self.current_row:
                self.current_table.append(self.current_row)
        elif tag == "table":
            if self.current_table:
                self.tables.append(self.current_table)
                self.current_table = []


def extract_html_tables(pp_data, paddle_data):
    html_blocks = []

    root_pp = get_root(pp_data)
    for block in root_pp.get("parsing_res_list", []):
        if block.get("block_label") == "table":
            content = block.get("block_content")
            if isinstance(content, str) and "<table" in content.lower():
                html_blocks.append(content)

    root_pd = get_root(paddle_data)
    for table_obj in root_pd.get("table_res_list", []):
        pred_html = table_obj.get("pred_html")
        if isinstance(pred_html, str) and "<table" in pred_html.lower():
            html_blocks.append(pred_html)

    all_tables = []
    for block in html_blocks:
        parser = SimpleHTMLTableParser()
        try:
            parser.feed(block)
            all_tables.extend(parser.tables)
        except Exception:
            continue

    return all_tables


def normalize_html_row_to_5cols(row):
    cells = [clean_text(c) for c in row if clean_text(c)]
    if not cells:
        return None

    joined = " ".join(cells)
    if looks_like_metadata_text(joined):
        return None
    if looks_like_header_text(joined):
        return None

    if len(cells) == 1:
        return [cells[0], "", "", "", ""]
    if len(cells) >= 5:
        return cells[:5]

    while len(cells) < 5:
        cells.append("")
    return cells


def build_rows_from_html(pp_data, paddle_data):
    tables = extract_html_tables(pp_data, paddle_data)
    best_rows = []

    for table in tables:
        candidate = []
        for row in table:
            normalized = normalize_html_row_to_5cols(row)
            if normalized is None:
                continue
            candidate.append(normalized)

        if len(candidate) > len(best_rows):
            best_rows = candidate

    return best_rows


def split_notes_from_rows(rows):
    table_rows = []
    note_rows = []

    for row in rows:
        joined = " ".join(row)
        if looks_like_note_anchor_text(joined):
            note_rows.append(joined)
            continue

        if row_has_description_and_value(row) or count_nonempty(row) >= 3:
            table_rows.append(row)
        else:
            note_rows.append(joined)

    return table_rows, note_rows


# ============================================================================
# THREE-TIER PARSING STRATEGY
# ============================================================================

def parse_classic_hematology(core_rows, tatr_box):
    """
    FIRST CASE: Classic hematology page
    Use strict parsing with fixed column boundaries for known analytes.
    """
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
    
    structured_rows = cleanup_bad_rows(structured_rows)
    return structured_rows


def split_generic_lab_row(row):
    """
    Smart splitter for generic 5-column lab rows.
    Keeps Description | Résultat | Unité | Valeurs normales | Val.
    
    Handles:
    - Oui/Non markers
    - Numeric results with +/- suffixes
    - Units separated from values
    - Normal ranges
    - Multiple test names in first column
    """
    row = [clean_text(c) for c in row]
    if len(row) != 5:
        return row
    
    desc, res, unit, ref, val = row
    
    # Case 1: No result but unit present → might be split wrong
    # Example: "Phosphatases Alcalines | | UI/137c | 120 400"
    # This is already in good format, keep as-is
    if not res and unit and not ref:
        return [desc, res, unit, ref, val]
    
    # Case 2: Result and unit merged in description or result
    # Example: "BilirubineTotale | 61+ umol/l | 0 17 | Bilirubine conjuguee"
    if desc and not unit and not res:
        # Try to split result from unit
        m = re.match(r"^(.*?)([-+]?\d+(?:[.,]\d+)?[\w\s]*?(?:umol|mg|g|UI|μ).*?)$", desc, re.IGNORECASE)
        if m:
            desc_part = clean_text(m.group(1))
            res_unit = clean_text(m.group(2))
            if desc_part and res_unit:
                # Further split result and unit
                res_unit_split = split_result_and_unit(res_unit)
                return [desc_part, res_unit_split[0], res_unit_split[1], ref, val]
    
    # Case 3: Result has both numeric and unit
    # Example: "46 | UI/137c | 5 60"
    if res and not unit:
        res_unit_split = split_result_and_unit(res)
        return [desc, res_unit_split[0], res_unit_split[1], unit, ref]
    
    # Case 4: Everything already split correctly
    return [desc, res, unit, ref, val]


def split_result_and_unit(result_str):
    """
    Split a result string like '61+ umol/l' or '46 UI/137c'
    into ['result', 'unit']
    """
    result_str = clean_text(result_str)
    
    # Try to find: number (with optional +/-) | unit
    m = re.match(r"^([-+]?\d+(?:[.,]\d+)?[\+\-]?)\s*(.*)$", result_str)
    if m:
        numeric = clean_text(m.group(1))
        unit_part = clean_text(m.group(2))
        return [numeric, unit_part]
    
    return [result_str, ""]


def parse_generic_lab_table(core_rows, tatr_box):
    """
    SECOND CASE: Generic 5-column lab table
    Use flexible row splitter for unknown analyte names.
    Separates numeric results from units and normal ranges.
    """
    bounds = detect_col_boundaries(tatr_box)
    structured_rows_ocr = []
    
    for row in core_rows:
        cols = [clean_text(c) for c in assign_cols(row, bounds)]
        joined = " ".join(cols).strip()
        
        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue
        
        # Apply smart splitting
        cols = split_generic_lab_row(cols)
        structured_rows_ocr.append(cols)
    
    structured_rows_ocr = cleanup_bad_rows(structured_rows_ocr)
    return structured_rows_ocr


def main():
    parser = argparse.ArgumentParser(description="Hybrid parser v2: Three-tier parsing strategy")
    parser.add_argument("pp_json")
    parser.add_argument("paddle_json")
    parser.add_argument("tatr_txt")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables",
    )
    parser.add_argument("--tatr-width", type=int, default=None)
    parser.add_argument("--tatr-height", type=int, default=None)
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

    tatr_size = get_tatr_render_size(Path(args.tatr_txt))
    if tatr_size is not None:
        tatr_render_w, tatr_render_h = tatr_size
    elif args.tatr_width and args.tatr_height:
        tatr_render_w, tatr_render_h = args.tatr_width, args.tatr_height
    else:
        tatr_render_w, tatr_render_h = 2502, 3516

    tatr_box = rescale_box(tatr_box_raw, tatr_render_w, tatr_render_h, pp_width, pp_height)
    lines = get_paddle_lines(paddle_data)

    upper_lines, core_lines, lower_lines = [], [], []
    for ln in lines:
        if not inside(ln, pp_box, pad=10):
            continue
        if ln["y2"] < tatr_box[1]:
            upper_lines.append(ln)
        elif inside(ln, tatr_box, pad=10):
            core_lines.append(ln)
        elif ln["y1"] > tatr_box[3]:
            lower_lines.append(ln)

    upper_rows = group_rows(upper_lines, y_thresh=18)
    upper_rows = [merge_close_in_row(r, x_gap=20) for r in upper_rows]
    upper_text = []
    for row in upper_rows:
        line = row_text(row)
        if line and not looks_like_metadata_text(line):
            upper_text.append(line)

    core_rows = group_rows(core_lines, y_thresh=18)
    core_rows = [merge_close_in_row(r, x_gap=20) for r in core_rows]

    # ========================================================================
    # THREE-TIER PARSING LOGIC
    # ========================================================================
    
    classic_hematology = is_classic_hematology_page_from_texts(lines)
    
    if classic_hematology:
        # FIRST CASE: Use classic hematology parsing
        structured_rows = parse_classic_hematology(core_rows, tatr_box)
        hematology_incomplete = is_hematology_ocr_incomplete(structured_rows)
        use_html_fallback = False
        
        # Only fallback if incomplete AND HTML is good
        if hematology_incomplete:
            structured_rows_html = build_rows_from_html(pp_data, paddle_data)
            html_table_rows, html_note_rows = split_notes_from_rows(structured_rows_html)
            html_table_rows = cleanup_bad_rows(html_table_rows)
            html_quality = structured_rows_quality(html_table_rows)
            
            if html_table_rows and html_quality >= 0.70:
                structured_rows = html_table_rows
                use_html_fallback = True
        
        ocr_quality = structured_rows_quality(structured_rows)
        html_quality = ocr_quality if not use_html_fallback else structured_rows_quality(
            build_rows_from_html(pp_data, paddle_data)
        )
    else:
        # SECOND CASE: Generic 5-column lab table with flexible splitter
        structured_rows = parse_generic_lab_table(core_rows, tatr_box)
        ocr_quality = structured_rows_quality(structured_rows)
        use_html_fallback = False
        
        # THIRD CASE: Only fallback if OCR quality too low
        if ocr_quality < 0.65:
            structured_rows_html = build_rows_from_html(pp_data, paddle_data)
            html_table_rows, html_note_rows = split_notes_from_rows(structured_rows_html)
            html_table_rows = cleanup_bad_rows(html_table_rows)
            html_quality = structured_rows_quality(html_table_rows)
            
            if html_table_rows and html_quality > ocr_quality:
                structured_rows = html_table_rows
                use_html_fallback = True
        else:
            html_quality = ocr_quality
        
        hematology_incomplete = False

    lower_rows = group_rows(lower_lines, y_thresh=18)
    lower_rows = [merge_close_in_row(r, x_gap=20) for r in lower_rows]
    lower_text = []
    for row in lower_rows:
        line = row_text(row)
        if line:
            lower_text.append(line)

    if use_html_fallback and not classic_hematology:
        html_table_rows, html_note_rows = split_notes_from_rows(build_rows_from_html(pp_data, paddle_data))
        for note in html_note_rows:
            if note and note not in lower_text:
                lower_text.append(note)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.pp_json).stem}_hybrid.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"PP box: {pp_box}\n")
        f.write(f"TATR raw box: {tatr_box_raw}\n")
        f.write(f"TATR scaled box: {tatr_box}\n")
        f.write(f"PP page size: {pp_width} x {pp_height}\n")
        f.write(f"TATR render size: {tatr_render_w} x {tatr_render_h}\n")
        f.write(f"OCR quality: {ocr_quality:.3f}\n")
        f.write(f"HTML quality: {html_quality:.3f}\n")
        f.write(f"Classic hematology page: {classic_hematology}\n")
        f.write(f"Hematology OCR incomplete: {hematology_incomplete}\n")
        f.write(f"Used HTML fallback: {use_html_fallback}\n\n")

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
    print(f"OCR quality: {ocr_quality:.3f}")
    print(f"HTML quality: {html_quality:.3f}")
    print(f"Classic hematology page: {classic_hematology}")
    print(f"Hematology OCR incomplete: {hematology_incomplete}")
    print(f"Used HTML fallback: {use_html_fallback}")


if __name__ == "__main__":
    main()
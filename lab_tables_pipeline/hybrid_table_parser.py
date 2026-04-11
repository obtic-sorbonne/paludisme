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

    x1 = min(int(b["block_bbox"][0]) for b in table_blocks)
    y1 = min(int(b["block_bbox"][1]) for b in table_blocks)
    x2 = max(int(b["block_bbox"][2]) for b in table_blocks)
    y2 = max(int(b["block_bbox"][3]) for b in table_blocks)

    return [x1, y1, x2, y2]


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


def resort_row_by_x(row):
    if not row:
        return row
    return sorted(row, key=lambda x: x["x1"])


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


def detect_col_boundaries_from_rows(rows, tatr_box, num_cols=5):
    if not rows:
        return None

    x1_box, _, x2_box, _ = tatr_box
    box_width = x2_box - x1_box

    x_positions = []
    for row in rows:
        for item in row:
            x_positions.append(item["x1"])
            x_positions.append(item["x2"])

    if not x_positions:
        return None

    x_positions = sorted(set(x_positions))

    gaps = []
    for i in range(len(x_positions) - 1):
        gap_size = x_positions[i + 1] - x_positions[i]
        if gap_size > 5:
            gap_center = (x_positions[i] + x_positions[i + 1]) / 2
            gaps.append((gap_center, gap_size))

    gaps.sort(key=lambda x: x[1], reverse=True)

    if len(gaps) >= 4:
        boundaries = sorted([g[0] for g in gaps[:4]])
    else:
        boundaries = [
            x1_box + 0.45 * box_width,
            x1_box + 0.62 * box_width,
            x1_box + 0.79 * box_width,
            x1_box + 0.94 * box_width,
        ]

    return boundaries


def assign_cols(row, bounds):
    if bounds is None:
        cols = [""] * 5
        sorted_items = sorted(row, key=lambda x: x["x1"])
        for i, item in enumerate(sorted_items[:5]):
            cols[i] = item["text"].strip()
        return cols

    cols = ["", "", "", "", ""]
    for item in row:
        x = item["cx"]
        text = item["text"].strip()

        idx = 0
        for i, bound in enumerate(bounds):
            if x >= bound:
                idx = i + 1
        idx = min(idx, 4)

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
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2",
        "page 2 sur 2", "page 1 sur 1", "hopital robert debre", "urgent"
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


def looks_like_short_medical_note(text):
    t = norm(clean_text(text))
    return t in {"negatif", "positif", "positive", "negative", "négatif", "positif"}


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
    joined = "\n".join(clean_text(x["text"]) for x in lines if clean_text(x["text"]))
    j = norm(joined)
    analytes = [
        "erythrocytes", "hemoglobine", "hematocrite", "leucocytes",
        "plaquettes", "lymphocytes", "monocytes", "polyneutrophiles"
    ]
    hits = sum(1 for a in analytes if a in j)
    return hits >= 5


def is_biochemistry_table_from_texts(lines):
    joined = "\n".join(clean_text(x["text"]) for x in lines if clean_text(x["text"]))
    j = norm(joined)

    analytes = [
        "hemolyse",
        "ictere",
        "lipemie",
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "proteines plasmatiques",
        "uree",
        "creatinine",
        "glycemie",
        "phosphatases alcalines",
        "bilirubine totale",
        "bilirubine conjuguee",
        "asat",
        "alat",
        "ggt",
        "prealbumine",
        "crp",
        "procalcitonine",
        "ldh",
        "haptoglobine",
    ]

    hits = sum(1 for a in analytes if a in j)

    if hits >= 3 and any(x in j for x in ["hemolyse", "ictere", "lipemie"]):
        return True

    return hits >= 5


def is_hematology_ocr_incomplete(rows):
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


def parse_classic_hematology(all_rows, tatr_box):
    bounds = detect_col_boundaries_from_rows(all_rows, tatr_box)

    raw_rows = []
    for row in all_rows:
        cols = [clean_text(c) for c in assign_cols(row, bounds)]
        joined = " ".join(cols).strip()

        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue

        raw_rows.append(cols)

    def is_section_title_row(cols):
        joined = norm(" ".join(cols))
        bad = [
            "examens d'hematologie",
            "examen d'hematologie",
            "cytologie",
            "description resultat unite valeurs normales val",
        ]
        return any(b in joined for b in bad)

    def is_header_garbage_row(cols):
        joined = norm(" ".join(cols))
        bad = [
            "copie:",
            "copie ",
            "st denis",
            "93200",
            "description resultat",
            "resultat unite",
            "valeurs normales",
            "val.",
        ]
        return any(b in joined for b in bad)

    def split_desc_value_if_merged(desc):
        desc = clean_text(desc)
        m = re.match(r"^(.*?)(?:\s+)([-+]?\d+(?:[.,]\d+)?[+-]?)$", desc)
        if m:
            left = clean_text(m.group(1))
            right = clean_text(m.group(2))
            if left and right:
                return left, right
        return desc, ""

    def normalize_desc_label(desc):
        d = clean_text(desc)
        dn = norm(d)

        if "reticulocytes" in dn and "%" in d:
            return "RETICULOCYTES %...."
        if "reticulocytes" in dn and "/mm3" in dn:
            return "RETICULOCYTES /mm3.."
        if "rech parasites sang" in dn:
            return "Rech parasites sang..."
        if "anomaliesmorph plaquette" in dn or "anomalies morph plaquette" in dn:
            return "Anomalies morph plaquette"
        if "anomaliesmorph leucocyte" in dn or "anomalies morph leucocyte" in dn:
            return "Anomalies morph leucocyte"
        return d

    def looks_like_valid_tail_label(desc):
        dn = norm(desc)
        allowed = [
            "reticulocytes",
            "rech parasites sang",
            "anomaliesmorph plaquette",
            "anomalies morph plaquette",
            "anomaliesmorph leucocyte",
            "anomalies morph leucocyte",
        ]
        return any(a in dn for a in allowed)

    repaired = []
    pending_label = None

    for cols in raw_rows:
        cols = [clean_text(c) for c in cols]
        desc, res, unit, ref, val = cols

        if is_section_title_row(cols):
            continue
        if is_header_garbage_row(cols):
            continue

        if desc and not res:
            new_desc, merged_value = split_desc_value_if_merged(desc)
            if merged_value:
                desc = new_desc
                res = merged_value
                cols = [desc, res, unit, ref, val]

        desc, res, unit, ref, val = cols
        joined = " ".join(cols).strip()

        if desc and not res and not unit and not ref:
            if looks_like_valid_tail_label(desc):
                if pending_label is not None:
                    repaired.append(pending_label)
                pending_label = [normalize_desc_label(desc), "", "", "", ""]
                continue

        if pending_label is not None:
            desc2, res2, unit2, ref2, val2 = cols
            has_number = bool(re.search(r"\d+[.,]\d+|\b\d+\b", joined))
            has_text_value = bool(re.search(r"\b(pos|neg|positive|negative|positif|negatif|ano|bl)\b", joined, re.I))
            has_unit = bool(re.search(r"%|/mm3", joined, re.I))

            no_real_desc = (not clean_text(desc2)) or (
                clean_text(desc2) and not bool(re.search(r"[A-Za-zÀ-ÿ]{3,}", desc2))
            )

            if no_real_desc and (has_number or has_text_value or has_unit):
                p_desc = pending_label[0]

                new_res = res2
                if not new_res:
                    m_num = re.search(r"([-+]?\d+(?:[.,]\d+)?[+-]?)", joined)
                    m_txt = re.search(r"\b(pos|neg|positive|negative|positif|negatif|ano|bl)\b", joined, re.I)
                    if m_num:
                        new_res = clean_text(m_num.group(1))
                    elif m_txt:
                        txt_val = clean_text(m_txt.group(1))
                        if norm(txt_val) in {"ano", "bl"}:
                            new_res = txt_val.upper()
                        else:
                            new_res = txt_val

                new_unit = unit2
                if not new_unit:
                    m_unit = re.search(r"(%|/mm3)", joined, re.I)
                    if m_unit:
                        new_unit = clean_text(m_unit.group(1))

                repaired.append([p_desc, new_res, new_unit, ref2, val2])
                pending_label = None
                continue

        if pending_label is not None:
            repaired.append(pending_label)
            pending_label = None

        repaired.append(cols)

    if pending_label is not None:
        repaired.append(pending_label)

    structured_rows = []
    for row in repaired:
        row = [clean_text(c) for c in row]
        joined = " ".join(row).strip()

        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue
        if is_section_title_row(row):
            continue
        if is_header_garbage_row(row):
            continue

        structured_rows.append(row)

    structured_rows = cleanup_bad_rows(structured_rows)

    final_rows = []
    i = 0
    while i < len(structured_rows):
        row = structured_rows[i]
        desc, res, unit, ref, val = row
        dn = norm(desc)

        if ("anomaliesmorph leucocyte" in dn or "anomalies morph leucocyte" in dn):
            value = clean_text(res)

            if not value and i + 1 < len(structured_rows):
                next_desc = clean_text(structured_rows[i + 1][0])
                if re.fullmatch(r"ANO|BL|ano|bl", next_desc):
                    value = next_desc.upper()
                    i += 1

            final_rows.append(["Anomalies morph leucocyte", value or "ANO", "", "", ""])
            i += 1
            continue

        if ("anomaliesmorph plaquette" in dn or "anomalies morph plaquette" in dn):
            value = clean_text(res)

            if not value and i - 1 >= 0:
                prev_desc = clean_text(structured_rows[i - 1][0])
                if re.fullmatch(r"ANO|BL|ano|bl", prev_desc):
                    value = prev_desc.upper()
                    if final_rows:
                        last_desc = clean_text(final_rows[-1][0])
                        if re.fullmatch(r"ANO|BL|ano|bl", last_desc):
                            final_rows.pop()

            if not value and i + 1 < len(structured_rows):
                next_desc = clean_text(structured_rows[i + 1][0])
                if re.fullmatch(r"ANO|BL|ano|bl", next_desc):
                    value = next_desc.upper()
                    i += 1

            final_rows.append(["Anomalies morph plaquette", value or "ANO", "", "", ""])
            i += 1
            continue

        if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?[+-]?", desc) and unit == "%":
            final_rows.append(["RETICULOCYTES %....", desc, "%", "", ""])
            i += 1
            continue

        if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?[+-]?", desc) and unit.lower() == "/mm3":
            final_rows.append(["RETICULOCYTES /mm3..", desc, "/mm3", "", ""])
            i += 1
            continue

        if re.fullmatch(r"(pos|neg|positive|negative|positif|negatif)", desc, re.I):
            final_rows.append(["Rech parasites sang...", desc, "", "", ""])
            i += 1
            continue

        if re.fullmatch(r"ANO|BL|ano|bl", desc) and not res and not unit:
            i += 1
            continue

        final_rows.append(row)
        i += 1

    dedup = []
    seen = set()
    for row in final_rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            dedup.append(row)

    return dedup


def normalize_biochem_result(text: str) -> str:
    t = clean_text(text)
    t = re.sub(r"^\s*<\s*", "<", t)
    t = re.sub(r"^\s*>\s*", ">", t)
    return t


def looks_like_biochem_result(text):
    t = normalize_biochem_result(text)
    return bool(re.fullmatch(
        r"(non|oui|pos|neg|positive|negative|positif|negatif|opa|[<>]?\s*[-+]?\d+(?:[.,]\d+)?[+-]?)",
        t,
        re.I
    ))


def looks_like_biochem_unit(text):
    t = norm(clean_text(text))
    return bool(re.search(
        r"(?:^|[\s(])("
        r"mmol/l|mmol/l\)|mml/l|mmo/l|mmol1l|mmol\\l|mmol\s*/\s*l|"
        r"umol/l|μmol/l|umol1l|umol\\l|umol\s*/\s*l|"
        r"g/l|g/1|g\\l|g\s*/\s*l|mg/l|mg/1|mg\\l|mg\s*/\s*l|"
        r"ug/l|μg/l|ug\\l|ug\s*/\s*l|"
        r"ui/l ?37c|ui/l37c|ui/137c|ui/l|u/l ?37c|u/l|l37c"
        r")(?:$|[\s)])",
        t,
        re.I
    ))


def looks_like_biochem_reference(text):
    t = clean_text(text)
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", t)
    return len(nums) >= 1


def normalize_biochem_unit(text):
    t = clean_text(text)
    tl = t.lower()

    tl = tl.replace(" ", "")
    tl = tl.replace("\\", "/")
    tl = tl.replace("1", "l")

    if "ui/l37c" in tl or "u/l37c" in tl or "ui/137c" in tl or "l37c" == tl:
        return "UI/L37c"
    if "ui/l" in tl or "u/l" in tl:
        return "UI/L"
    if "umol/l" in tl or "μmol/l" in tl:
        return "umol/L"
    if "ug/l" in tl or "μg/l" in tl:
        return "ug/L"
    if "mmol/l" in tl or "mml/l" in tl or "mmo/l" in tl:
        return "mmol/L"
    if "mg/l" in tl:
        return "mg/L"
    if "g/l" in tl:
        return "g/L"

    return clean_text(text)


def split_reference_range(ref_text):
    ref_text = clean_text(ref_text)
    nums = re.findall(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?", ref_text)
    if len(nums) >= 2:
        return clean_text(nums[0]), clean_text(nums[1])
    if len(nums) == 1:
        return clean_text(nums[0]), ""
    return "", ""


def text_has_numeric_value(text: str) -> bool:
    return bool(re.search(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?[+-]?", clean_text(text)))


def text_has_qualitative_value(text: str) -> bool:
    return bool(re.search(
        r"\b(non|oui|pos|neg|positive|negative|positif|negatif|opa)\b",
        clean_text(text),
        re.I,
    ))


def text_has_unit_token(text: str) -> bool:
    return looks_like_biochem_unit(text)


def text_has_reference_pair(text: str) -> bool:
    nums = re.findall(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?", clean_text(text))
    return len(nums) >= 1


def row_has_real_biochem_payload(result, unit, ref_min, ref_max):
    if clean_text(result):
        return True
    if clean_text(unit):
        return True
    if clean_text(ref_min) or clean_text(ref_max):
        return True
    return False


def row_can_receive_unit(row):
    desc = clean_text(row.get("desc", ""))
    result = clean_text(row.get("result", ""))
    dn = norm(desc)

    if not desc:
        return False
    if dn in {"hemolyse", "ictere", "lipemie"}:
        return False
    if not result:
        return False

    return True


def row_can_receive_reference(row):
    desc = clean_text(row.get("desc", ""))
    result = clean_text(row.get("result", ""))
    unit = clean_text(row.get("unit", ""))
    dn = norm(desc)

    if not desc:
        return False
    if dn in {"hemolyse", "ictere", "lipemie"}:
        return False
    if not result and not unit:
        return False

    return True


def looks_like_biochem_qual_result(text: str) -> bool:
    t = norm(clean_text(text))
    return t in {"non", "oui", "opa", "neg", "pos", "negative", "positive", "positif", "negatif"}


def looks_like_biochem_numeric_result(text: str) -> bool:
    t = normalize_biochem_result(text)
    return bool(re.fullmatch(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?[+-]?", t))


def looks_like_sparse_biochem_note(desc: str, result: str, unit: str, ref_min: str, ref_max: str, desc_x1=None, base_desc_x1=None) -> bool:
    d = clean_text(desc)
    dn = norm(d)
    r = clean_text(result)
    u = clean_text(unit)
    a = clean_text(ref_min)
    b = clean_text(ref_max)

    if not d:
        return True

    comment_phrases = [
        "resultat controle",
        "resultat controlé",
        "resultat telephone",
        "resultat téléphoné",
        "antibiotherapie",
        "antibiotherapie (oui/non)",
        "opalescent",
        "valide par",
        "validé par",
    ]
    if any(p in dn for p in comment_phrases):
        return True

    if desc_x1 is not None and base_desc_x1 is not None:
        if desc_x1 > base_desc_x1 + 25 and not r and not u and not a and not b:
            return True

    if not r and not u and not a and not b:
        return True

    return False


def is_real_biochem_row(desc: str, result: str, unit: str, ref_min: str, ref_max: str) -> bool:
    d = clean_text(desc)
    r = clean_text(result)
    u = clean_text(unit)
    a = clean_text(ref_min)
    b = clean_text(ref_max)

    if not d:
        return False

    if r and looks_like_biochem_qual_result(r):
        return True

    if r and looks_like_biochem_numeric_result(r):
        if u or a or b:
            return True
        return True

    if u and (a or b):
        return True

    return False


def parse_biochemistry_lab_table(all_lines, tatr_box):
    """
    Structural biochemistry parser.

    Important:
    - rows are NOT kept only because their label is known
    - a row must have its own payload (result/unit/ref) to stay in the table
    - label-only rows become notes
    - units are attached by row-local geometry, not hardcoded
    - reference ranges are attached by row-local geometry first, then soft fallback
    - preserves < and > in both results and reference values
    """

    def normalize_biochem_ref_token(text: str) -> str:
        t = clean_text(text)
        t = re.sub(r"^\s*<\s*", "<", t)
        t = re.sub(r"^\s*>\s*", ">", t)
        return t

    def extract_ref_tokens(text: str):
        return [
            normalize_biochem_ref_token(x)
            for x in re.findall(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?", clean_text(text))
        ]

    x1, y1, x2, y2 = tatr_box
    w = x2 - x1

    desc_right = x1 + 0.38 * w
    result_left = desc_right
    result_right = x1 + 0.53 * w
    unit_left = x1 + 0.53 * w
    unit_right = x1 + 0.72 * w
    ref_left = x1 + 0.72 * w
    ref_right = x1 + 1.05 * w   # slightly wider than before to catch rightmost ref values

    lines = []
    for ln in all_lines:
        if inside(ln, tatr_box, pad=35):
            txt = clean_text(ln["text"])
            if not txt:
                continue
            if looks_like_metadata_text(txt):
                continue
            if looks_like_header_text(txt):
                continue
            lines.append(ln)

    if not lines:
        return [], []

    desc_items = []
    payload_items = []

    for ln in lines:
        if ln["cx"] <= desc_right:
            desc_items.append(ln)
        else:
            payload_items.append(ln)

    desc_rows = group_rows(desc_items, y_thresh=16)
    desc_rows = [merge_close_in_row(r, x_gap=20) for r in desc_rows]
    desc_rows = [resort_row_by_x(r) for r in desc_rows]

    candidate_rows = []
    for row in desc_rows:
        desc_text = clean_text(" ".join(x["text"] for x in row))
        if not desc_text:
            continue
        if looks_like_metadata_text(desc_text):
            continue
        if looks_like_header_text(desc_text):
            continue

        desc_norm = norm(desc_text)
        if any(bad in desc_norm for bad in [
            "biochimie generale",
            "examens de sang",
            "description resultat unite valeurs normales val",
            "valide par",
        ]):
            continue

        desc_text = desc_text.replace("BilirubineTotale", "Bilirubine Totale")
        desc_text = desc_text.replace("ProteinesPlasmatiques", "Proteines Plasmatiques")

        cy = sum(x["cy"] for x in row) / len(row)

        candidate_rows.append({
            "desc": clean_text(desc_text),
            "cy": cy,
            "result": "",
            "unit": "",
            "ref_min": "",
            "ref_max": "",
        })

    note_rows = []

    # Step 1: row-local result extraction
    for row in candidate_rows:
        cy = row["cy"]
        nearby = [item for item in payload_items if abs(item["cy"] - cy) <= 16]
        nearby = sorted(nearby, key=lambda x: x["x1"])

        result = ""

        for item in nearby:
            txt = clean_text(item["text"])
            if not txt:
                continue

            cx = item["cx"]
            if result_left <= cx <= result_right:
                if looks_like_biochem_result(txt):
                    result = normalize_biochem_result(txt)
                    break

                m = re.search(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?[+-]?", txt)
                if m:
                    result = normalize_biochem_result(m.group(0))
                    break

        if not result:
            joined_nearby = " ".join(
                clean_text(x["text"])
                for x in nearby
                if result_left <= x["cx"] <= result_right
            )
            if joined_nearby:
                m_res = re.search(
                    r"\b(non|oui|pos|neg|positive|negative|positif|negatif|opa)\b",
                    joined_nearby,
                    re.I,
                )
                if m_res:
                    result = clean_text(m_res.group(1))
                else:
                    m_num = re.search(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?[+-]?", joined_nearby)
                    if m_num:
                        result = normalize_biochem_result(m_num.group(0))

        row["result"] = result

    # Step 2: row-local unit extraction
    for row in candidate_rows:
        if not row_can_receive_unit(row):
            continue

        cy = row["cy"]
        nearby_units = [
            item for item in payload_items
            if abs(item["cy"] - cy) <= 16 and unit_left <= item["cx"] <= unit_right
        ]
        nearby_units = sorted(nearby_units, key=lambda x: x["x1"])

        unit_val = ""
        for item in nearby_units:
            txt = clean_text(item["text"])
            if not txt:
                continue
            if looks_like_biochem_unit(txt):
                unit_val = normalize_biochem_unit(txt)
                break

        if not unit_val:
            joined_units = " ".join(clean_text(x["text"]) for x in nearby_units)
            m = re.search(
                r"(mmol/l|mmol\s*/\s*l|mml/l|mmo/l|umol/l|μmol/l|g/l|g\s*/\s*l|mg/l|mg\s*/\s*l|ug/l|µg/l|ui/l ?37c|ui/l37c|ui/137c|ui/l|u/l ?37c|u/l|l37c)",
                joined_units,
                re.I,
            )
            if m:
                unit_val = normalize_biochem_unit(m.group(1)).replace("ug/L", "ug/L")

        row["unit"] = clean_text(unit_val)

    # Step 3: row-local reference extraction
    for row in candidate_rows:
        if not row_can_receive_reference(row):
            continue

        cy = row["cy"]
        nearby_refs = [
            item for item in payload_items
            if abs(item["cy"] - cy) <= 18 and ref_left <= item["cx"] <= ref_right
        ]
        nearby_refs = sorted(nearby_refs, key=lambda x: x["x1"])

        ref_tokens = []
        for item in nearby_refs:
            txt = clean_text(item["text"])
            if not txt:
                continue
            ref_tokens.extend(extract_ref_tokens(txt))

        if len(ref_tokens) >= 2:
            row["ref_min"] = clean_text(ref_tokens[0])
            row["ref_max"] = clean_text(ref_tokens[1])
        elif len(ref_tokens) == 1:
            row["ref_min"] = clean_text(ref_tokens[0])

        # Row-joined retry for split OCR fragments
        if not (row["ref_min"] and row["ref_max"]):
            joined_refs = " ".join(clean_text(x["text"]) for x in nearby_refs)
            joined_tokens = extract_ref_tokens(joined_refs)

            if len(joined_tokens) >= 2:
                row["ref_min"] = clean_text(joined_tokens[0])
                row["ref_max"] = clean_text(joined_tokens[1])
            elif len(joined_tokens) == 1 and not row["ref_min"]:
                row["ref_min"] = clean_text(joined_tokens[0])

    # Step 4: global soft fallback for missing references only
    ref_items = []
    for ln in lines:
        if ref_left <= ln["cx"] <= ref_right:
            txt = clean_text(ln["text"])
            if not txt:
                continue
            tokens = extract_ref_tokens(txt)
            if tokens:
                ref_items.append(ln)

    ref_rows = group_rows(ref_items, y_thresh=14)
    ref_rows = [merge_close_in_row(r, x_gap=25) for r in ref_rows]
    ref_rows = [resort_row_by_x(r) for r in ref_rows]

    global_ref_pairs = []
    for row in ref_rows:
        row_txt = " ".join(clean_text(x["text"]) for x in row)
        toks = extract_ref_tokens(row_txt)
        if len(toks) >= 2:
            global_ref_pairs.append((toks[0], toks[1]))
        elif len(toks) == 1:
            global_ref_pairs.append((toks[0], ""))

    merged_ref_pairs = []
    i = 0
    while i < len(global_ref_pairs):
        a, b = global_ref_pairs[i]
        if a and b:
            merged_ref_pairs.append((a, b))
            i += 1
            continue

        if a and not b and i + 1 < len(global_ref_pairs):
            c, d = global_ref_pairs[i + 1]
            if c and not d:
                merged_ref_pairs.append((a, c))
                i += 2
                continue

        merged_ref_pairs.append((a, b))
        i += 1

    analyte_need_ref_fallback = []
    for idx, row in enumerate(candidate_rows):
        if row_can_receive_reference(row) and not (clean_text(row["ref_min"]) and clean_text(row["ref_max"])):
            analyte_need_ref_fallback.append(idx)

    for idx_row, ref_pair in zip(analyte_need_ref_fallback, merged_ref_pairs):
        if not candidate_rows[idx_row]["ref_min"]:
            candidate_rows[idx_row]["ref_min"] = clean_text(ref_pair[0])
        if not candidate_rows[idx_row]["ref_max"]:
            candidate_rows[idx_row]["ref_max"] = clean_text(ref_pair[1])

    # Step 5: split validated table rows vs notes
    validated_rows = []
    for row in candidate_rows:
        if row_has_real_biochem_payload(row["result"], row["unit"], row["ref_min"], row["ref_max"]):
            validated_rows.append(row)
        else:
            note_rows.append(row["desc"])

    # Step 6: final cleanup
    final_rows = []
    for row in validated_rows:
        desc = clean_text(row["desc"])
        result = clean_text(row["result"])
        unit = clean_text(row["unit"])
        ref_min = clean_text(row["ref_min"])
        ref_max = clean_text(row["ref_max"])

        joined = " ".join([desc, result, unit, ref_min, ref_max]).strip()
        if not desc:
            continue
        if looks_like_metadata_text(joined):
            continue
        if looks_like_header_text(joined):
            continue

        desc_norm = norm(desc)
        if any(bad in desc_norm for bad in [
            "biochimie generale",
            "examens de sang",
            "valide par",
        ]):
            continue

        if result or unit or ref_min or ref_max:
            final_rows.append([desc, result, unit, ref_min, ref_max])

    dedup_rows = []
    seen = set()
    for row in final_rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            dedup_rows.append(row)

    dedup_notes = []
    seen_notes = set()
    for note in note_rows:
        note = clean_text(note)
        if not note:
            continue
        if note not in seen_notes:
            seen_notes.add(note)
            dedup_notes.append(note)

    return dedup_rows, dedup_notes



def parse_generic_lab_table(core_rows, tatr_box):
    bounds = detect_col_boundaries_from_rows(core_rows, tatr_box)
    structured_rows_ocr = []

    for row in core_rows:
        cols = [clean_text(c) for c in assign_cols(row, bounds)]
        joined = " ".join(cols).strip()

        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue

        structured_rows_ocr.append(cols)

    structured_rows_ocr = cleanup_bad_rows(structured_rows_ocr)
    return structured_rows_ocr


def repair_morphology_rows_from_notes(table_rows, note_lines):
    repaired_notes = []
    found_plaquette_label = False
    found_plaquette_value = None

    for idx, line in enumerate(note_lines):
        t = clean_text(line)
        nt = norm(t)

        if "anomaliesmorph plaquette" in nt or "anomalies morph plaquette" in nt:
            found_plaquette_label = True

            if re.search(r"\bANO\b", t, re.I):
                found_plaquette_value = "ANO"
            elif re.search(r"\bBL\b", t, re.I):
                found_plaquette_value = "BL"
            elif idx + 1 < len(note_lines):
                nxt = clean_text(note_lines[idx + 1])
                if re.fullmatch(r"ANO|BL|ano|bl", nxt):
                    found_plaquette_value = nxt.upper()
            elif idx - 1 >= 0:
                prv = clean_text(note_lines[idx - 1])
                if re.fullmatch(r"ANO|BL|ano|bl", prv):
                    found_plaquette_value = prv.upper()

            continue

        repaired_notes.append(t)

    inserted = False
    for row in table_rows:
        desc = clean_text(row[0])
        if "anomaliesmorph plaquette" in norm(desc) or "anomalies morph plaquette" in norm(desc):
            row[0] = "Anomalies morph plaquette"
            if not clean_text(row[1]) and found_plaquette_value:
                row[1] = found_plaquette_value
            inserted = True
            break

    if found_plaquette_label and not inserted:
        table_rows.append([
            "Anomalies morph plaquette",
            found_plaquette_value or "ANO",
            "",
            "",
            ""
        ])

    return table_rows, repaired_notes


def preserve_short_medical_notes(note_lines):
    kept = []
    for line in note_lines:
        t = clean_text(line)
        nt = norm(t)

        if nt in {"negatif", "positif", "negative", "positive"}:
            kept.append(t)
            continue

        if t:
            kept.append(t)

    return kept


def main():
    parser = argparse.ArgumentParser(description="Hybrid parser with adaptive column boundary detection")
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

    classic_hematology = is_classic_hematology_page_from_texts(lines)
    biochemistry_table = is_biochemistry_table_from_texts(lines)

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

    if classic_hematology:
        extra_lower = []
        for ln in lines:
            txt = clean_text(ln["text"])
            if not txt:
                continue
            if ln["y1"] <= tatr_box[3]:
                continue
            if ln["x1"] < (tatr_box[0] - 180) or ln["x2"] > (tatr_box[2] + 180):
                continue
            if looks_like_metadata_text(txt):
                continue
            extra_lower.append(ln)

        seen = {(x["text"], x["x1"], x["y1"], x["x2"], x["y2"]) for x in lower_lines}
        for ln in extra_lower:
            key = (ln["text"], ln["x1"], ln["y1"], ln["x2"], ln["y2"])
            if key not in seen:
                lower_lines.append(ln)
                seen.add(key)

    upper_rows = group_rows(upper_lines, y_thresh=18)
    upper_rows = [merge_close_in_row(r, x_gap=20) for r in upper_rows]
    upper_rows = [resort_row_by_x(r) for r in upper_rows]
    upper_text = []
    for row in upper_rows:
        line = row_text(row)
        if line and not looks_like_metadata_text(line):
            upper_text.append(line)

    core_rows = group_rows(core_lines, y_thresh=18)
    core_rows = [merge_close_in_row(r, x_gap=20) for r in core_rows]
    core_rows = [resort_row_by_x(r) for r in core_rows]

    biochem_note_rows = []

    if classic_hematology:
        classic_rows = upper_rows + core_rows
        classic_rows = sorted(
            classic_rows,
            key=lambda r: sum(x["cy"] for x in r) / max(len(r), 1)
        )

        structured_rows = parse_classic_hematology(classic_rows, tatr_box)

        hematology_incomplete = is_hematology_ocr_incomplete(structured_rows)
        use_html_fallback = False

        if hematology_incomplete:
            structured_rows_html = build_rows_from_html(pp_data, paddle_data)
            html_table_rows, html_note_rows = split_notes_from_rows(structured_rows_html)
            html_table_rows = cleanup_bad_rows(html_table_rows)
            html_quality = structured_rows_quality(html_table_rows)

            if html_table_rows and html_quality >= 0.70:
                structured_rows = html_table_rows
                use_html_fallback = True

        ocr_quality = structured_rows_quality(structured_rows)
        html_quality = (
            ocr_quality
            if not use_html_fallback
            else structured_rows_quality(build_rows_from_html(pp_data, paddle_data))
        )

    elif biochemistry_table:
        biochem_lines = upper_lines + core_lines
        biochem_lines = sorted(biochem_lines, key=lambda x: (x["cy"], x["x1"]))

        structured_rows, biochem_note_rows = parse_biochemistry_lab_table(biochem_lines, tatr_box)
        ocr_quality = structured_rows_quality(structured_rows)
        use_html_fallback = False
        hematology_incomplete = False
        html_quality = ocr_quality

    else:
        structured_rows = parse_generic_lab_table(core_rows, tatr_box)
        ocr_quality = structured_rows_quality(structured_rows)
        use_html_fallback = False

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

    protected_notes = []
    for ln in lower_lines:
        txt = clean_text(ln["text"])
        if looks_like_short_medical_note(txt):
            protected_notes.append(txt)

    for note in protected_notes:
        if note not in lower_text:
            lower_text.append(note)

    if biochem_note_rows:
        for note in biochem_note_rows:
            note = clean_text(note)
            if note and note not in lower_text:
                lower_text.append(note)

    if use_html_fallback and not classic_hematology:
        html_table_rows, html_note_rows = split_notes_from_rows(build_rows_from_html(pp_data, paddle_data))
        for note in html_note_rows:
            if note and note not in lower_text:
                lower_text.append(note)

    structured_rows, lower_text = repair_morphology_rows_from_notes(structured_rows, lower_text)
    lower_text = preserve_short_medical_notes(lower_text)

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
        f.write(f"Biochemistry table page: {biochemistry_table}\n")
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
    print(f"Biochemistry table page: {biochemistry_table}")
    print(f"Hematology OCR incomplete: {hematology_incomplete}")
    print(f"Used HTML fallback: {use_html_fallback}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

from paddleocr import PaddleOCR

from table_reconstruction import parse_table_box_generic, format_parsed_table


def clean_line(line: str) -> str:
    line = str(line).replace("\xa0", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def fix_common_ocr_errors(line: str) -> str:
    line = clean_line(line)

    replacements = [
        ("Personnè à prévenir", "Personne à prévenir"),
        ("PA:I", "PA : /"),
        ("PA:l", "PA : /"),
        ("PA: I", "PA : /"),
        ("PA BrasG : I", "PA BrasG : /"),
        ("PA BrasG:I", "PA BrasG : /"),
        ("PA BrasG: I", "PA BrasG : /"),
        ("Actions lA0", "Actions IAO"),
        ("Actions lAO", "Actions IAO"),
        ("Actions IA0", "Actions IAO"),
        ("na parle pas", "ne parle pas"),
        ("Côte d'lvoire", "Côte d'Ivoire"),
        ("Cote d'lvoire", "Cote d'Ivoire"),
        ("momis", "mois"),
        ("oû", "où"),
        ("LARiAM", "LARIAM"),
        ("Personnè", "Personne"),
        ("Caret de santé présenté", "Carnet de santé présenté"),
    ]

    for old, new in replacements:
        line = line.replace(old, new)

    line = re.sub(r"\bIA0\b", "IAO", line)
    line = re.sub(r"\blA0\b", "IAO", line)
    line = re.sub(r"\blAO\b", "IAO", line)
    line = re.sub(r"\bd['’]lvoire\b", "d'Ivoire", line)

    line = re.sub(r"\s+:", " :", line)
    line = re.sub(r"\s+;", ";", line)

    return line


def postprocess_page_lines(lines):
    cleaned = []
    prev = None

    for line in lines:
        line = fix_common_ocr_errors(line)
        if not line:
            continue
        if line == prev:
            continue
        cleaned.append(line)
        prev = line

    return cleaned


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(d: dict):
    return d.get("res", d)


def extract_tokens_from_page_result(page) -> list[dict]:
    rec_texts = []
    rec_boxes = []

    if isinstance(page, dict):
        if "rec_texts" in page and "rec_boxes" in page:
            rec_texts = page.get("rec_texts", [])
            rec_boxes = page.get("rec_boxes", [])
        else:
            root = get_root(page)
            ocr = root.get("overall_ocr_res", {})
            rec_texts = ocr.get("rec_texts", [])
            rec_boxes = ocr.get("rec_boxes", [])

    tokens = []
    
    print("[DEBUG] rec_texts =", len(rec_texts), "rec_boxes =", len(rec_boxes))
    if len(rec_boxes) > 0:
        sample_box = rec_boxes[0]
        print("[DEBUG] sample box type =", type(sample_box))
        print("[DEBUG] sample box value =", sample_box)

    for text, box in zip(rec_texts, rec_boxes):
        if not isinstance(text, str):
            continue

        text = fix_common_ocr_errors(text)
        if not text:
            continue

        x1 = y1 = x2 = y2 = None

        try:
            box_list = box.tolist() if hasattr(box, "tolist") else box
        except Exception:
            box_list = box

        # Case 1: [x1, y1, x2, y2]
        if (
            hasattr(box_list, "__len__")
            and len(box_list) == 4
            and all(isinstance(v, (int, float)) for v in box_list)
        ):
            x1, y1, x2, y2 = [float(v) for v in box_list]

        # Case 2: [[x,y], [x,y], [x,y], [x,y]]
        elif (
            hasattr(box_list, "__len__")
            and len(box_list) == 4
            and all(hasattr(pt, "__len__") and len(pt) >= 2 for pt in box_list)
        ):
            xs = [float(pt[0]) for pt in box_list]
            ys = [float(pt[1]) for pt in box_list]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Case 3: [x1,y1,x2,y2,x3,y3,x4,y4]
        elif (
            hasattr(box_list, "__len__")
            and len(box_list) == 8
            and all(isinstance(v, (int, float)) for v in box_list)
        ):
            xs = [float(box_list[i]) for i in range(0, 8, 2)]
            ys = [float(box_list[i]) for i in range(1, 8, 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        if x1 is None:
            continue

        tokens.append(
            {
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )

    return tokens


def extract_lines_from_page_result(page) -> list[str]:
    page_lines = []

    if isinstance(page, dict):
        rec_texts = page.get("rec_texts", [])
        if rec_texts:
            for line in rec_texts:
                page_lines.append(str(line))
        else:
            root = get_root(page)
            ocr = root.get("overall_ocr_res", {})
            rec_texts = ocr.get("rec_texts", [])
            if rec_texts:
                for line in rec_texts:
                    page_lines.append(str(line))
            else:
                page_lines.append(str(page))
    else:
        page_lines.append(str(page))

    return postprocess_page_lines(page_lines)


def get_table_blocks_from_pp_json(pp_json_path: Path):
    if not pp_json_path.exists():
        return []

    data = load_json(pp_json_path)
    root = get_root(data)
    blocks = root.get("parsing_res_list", [])

    table_blocks = []
    for block in blocks:
        if block.get("block_label") != "table":
            continue

        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        table_blocks.append((x1, y1, x2, y2))

    return table_blocks


def token_inside_box(tok, box, pad=3):
    x1, y1, x2, y2 = box
    return (
        tok["x1"] >= x1 - pad
        and tok["y1"] >= y1 - pad
        and tok["x2"] <= x2 + pad
        and tok["y2"] <= y2 + pad
    )


def remove_tokens_in_box(tokens, box, pad=3):
    kept = []
    removed = []
    for tok in tokens:
        if token_inside_box(tok, box, pad=pad):
            removed.append(tok)
        else:
            kept.append(tok)
    return kept, removed


def looks_table_like(parsed_table: dict) -> bool:
    if not parsed_table:
        return False
    rows = parsed_table.get("rows", [])
    dates = parsed_table.get("dates", [])
    return len(rows) >= 2 and len(dates) >= 2

def format_parsed_table(title_lines, parsed_table):
    out = []

    # keep clean unique titles
    seen_titles = set()
    for t in title_lines:
        tt = clean_line(t)
        if not tt:
            continue
        if tt not in seen_titles:
            out.append(tt)
            seen_titles.add(tt)

    if out:
        out.append("")

    rows = parsed_table.get("rows", [])
    notes = parsed_table.get("notes", [])
    review_needed = parsed_table.get("review_needed", False)
    review_reasons = parsed_table.get("review_reasons", [])

    for row in rows:
        label = clean_line(row.get("label", ""))
        if not label:
            continue

        values = row.get("values", {})
        if not values:
            continue

        max_col = max(values.keys()) if values else -1
        ordered_vals = []
        for col_idx in range(max_col + 1):
            val = values.get(col_idx)
            ordered_vals.append(clean_line(val) if val else "-")

        out.append(f"{label}: {' | '.join(ordered_vals)}")

    if notes:
        cleaned_notes = []
        seen_notes = set()
        for n in notes:
            nn = clean_line(n)
            if nn and nn not in seen_notes and not is_junk_footer_line(nn):
                seen_notes.add(nn)
                cleaned_notes.append(nn)

        if cleaned_notes:
            out.append("")
            out.append("Unassigned values / notes: " + " | ".join(cleaned_notes))

    if review_needed:
        out.append("")
        out.append("TABLE_REVIEW_NEEDED")
        for reason in review_reasons:
            out.append(f"- {clean_line(reason)}")

    return out

def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR narrative extraction with PP table reconstruction")
    parser.add_argument("pdf_path", help="Full path to the PDF file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle",
        help="Directory where output text file will be saved",
    )
    parser.add_argument(
        "--pp-json-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/pp_structure",
        help="Directory containing PP-Structure page JSONs",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    out_dir = Path(args.output_dir)
    pp_json_dir = Path(args.pp_json_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="fr",
    )

    result = ocr.predict(str(pdf_path))
    out_file = out_dir / f"{pdf_path.stem}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for page_idx, page in enumerate(result, start=1):
            f.write(f"===== PAGE {page_idx} =====\n\n")

            page_lines = extract_lines_from_page_result(page)
            page_tokens = extract_tokens_from_page_result(page)
            print(f"[DEBUG] page {page_idx}: total extracted page tokens = {len(page_tokens)}")
            pp_json_path = pp_json_dir / f"{pdf_path.stem}_{page_idx-1}_res.json"
            table_blocks = get_table_blocks_from_pp_json(pp_json_path)

            parsed_tables_output = []

            if table_blocks:
                print(f"[DEBUG] page {page_idx}: using PP JSON = {pp_json_path}")
                print(f"[DEBUG] table blocks found: {len(table_blocks)}")
            else:
                print(f"[DEBUG] page {page_idx}: no PP table blocks found")

            remaining_tokens = page_tokens[:]

            for box_idx, box in enumerate(table_blocks, start=1):
                remaining_tokens, tokens_in_box = remove_tokens_in_box(remaining_tokens, box, pad=3)

                print(f"[DEBUG] box {box_idx}: {box}")
                print(f"[DEBUG] box {box_idx}: tokens in box = {len(tokens_in_box)}")

                if not tokens_in_box:
                    print(f"[DEBUG] box {box_idx}: parsed = NO (no tokens)")
                    continue

                parsed = parse_table_box_generic(tokens_in_box)
                if parsed and looks_table_like(parsed):
                    print(f"[DEBUG] box {box_idx}: parsed = YES")

                    title_lines = []

                    # a very simple heading guess from nearby page lines
                    for i, line in enumerate(page_lines):
                        if "NFS" in line or "ionogramme" in line.lower() or "bilan hépatique" in line.lower():
                            title_lines = [line]
                            if i > 0 and page_lines[i - 1].isupper():
                                title_lines.insert(0, page_lines[i - 1])
                            break

                    parsed_tables_output.extend(format_parsed_table(title_lines, parsed))
                else:
                    print(f"[DEBUG] box {box_idx}: parsed = NO")

            for line in page_lines:
                f.write(line + "\n")

            if parsed_tables_output:
                f.write("\n===== REFORMATTED TABLES =====\n\n")
                for line in parsed_tables_output:
                    f.write(line + "\n")

            f.write("\n")
            print(f"Page {page_idx}: extracted {len(page_lines)} line(s)")

    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
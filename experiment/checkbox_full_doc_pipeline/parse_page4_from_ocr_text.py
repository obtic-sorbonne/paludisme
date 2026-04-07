from pathlib import Path
import argparse
import json
import re

from cnr_common import (
    load_ocr_txt,
    extract_page_block,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    norm,
    clean_text,
    parse_prefix_and_text,
    find_first_line_index,
    is_lone_marker,
    line_select_state,
)


def detect_inline_x_option(line: str, options: list[str]):
    raw = clean_text(line)
    raw_norm = norm(raw)
    for opt in options:
        opt_norm = norm(opt)
        if f"x {opt_norm}" in raw_norm or raw_norm.startswith(f"x {opt_norm}"):
            return opt
    return None


def page4_specs():
    return [
        {
            "field": "Utilisation traitement à visée Curative du paludisme dans les 30 derniers jours",
            "single_choice": True,
            "start_anchors": ["Utilisation traitement à visée Curative du paludisme dans les 30 derniers jours"],
            "end_anchors": ["Traitement :", "Prise en charge & traitement"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Prise en charge",
            "single_choice": False,
            "start_anchors": ["Prise en charge & traitement"],
            "end_anchors": ["Evolution clinique"],
            "max_lines": 15,
            "options": [
                ("Ambulatoire", ["Ambulatoire"]),
                ("Hospitalisation", ["Hospitalisation"]),
                ("Transfert autre hôpital", ["Transfert autre hôpital", "Transfert autre hopital"]),
                ("Pas de traitement", ["Pas de traitement"]),
            ],
        },
        {
            "field": "Evolution clinique",
            "single_choice": True,
            "start_anchors": ["Evolution clinique"],
            "end_anchors": ["Poids", "Médicaments - Traitement", "Traitement anti-palustre"],
            "max_lines": 8,
            "options": [
                ("Guérison", ["Guérison", "Guerison"]),
                ("DECES", ["DECES"]),
            ],
        },
        {
            "field": "Effet indésirable 1ère intention",
            "single_choice": True,
            "start_anchors": ["Effet indésirable", "Effet indesirable"],
            "end_anchors": ["Traitement débuté le", "Traitement debute le"],
            "max_lines": 8,
            "options": [
                ("Non", ["Non"]),
                ("Mineur", ["Mineur"]),
                ("Grave", ["Grave"]),
            ],
        },
    ]


# --------------------------------------------------
# Custom parser: Contrôle parasitologique (text-based)
# --------------------------------------------------

def parse_controle_parasitologique(lines):
    row_names = ["J3 ou J4", "J7 +/-1", "J28 +/-2", "Autre"]

    result = {
        "field": "Contrôle parasitologique P falciparum",
        "found": False,
        "control_overall": None,
        "rows": []
    }

    start_idx = None
    for i, line in enumerate(lines):
        if "controle parasitologique p falciparum" in norm(line):
            start_idx = i
            break

    if start_idx is None:
        return result

    result["found"] = True
    block = lines[start_idx:start_idx + 50]

    # Overall Oui / Non — check first ~6 lines of block
    for i in range(min(6, len(block))):
        prefix, content = parse_prefix_and_text(block[i])
        txt_norm = norm(content if prefix else block[i])

        if prefix == "X" and txt_norm in {"oui", "non"}:
            result["control_overall"] = content.capitalize() if content.lower() in {"oui", "non"} else content
            break
        # Also handle orphaned marker + next line pattern
        if prefix == "X" and i + 1 < len(block):
            next_norm = norm(block[i + 1])
            if next_norm in {"oui", "non"}:
                result["control_overall"] = block[i + 1].strip().capitalize()
                break

    # Row positions
    row_positions = []
    for rn in row_names:
        idx = None
        for i, line in enumerate(block):
            if norm(rn) in norm(line):
                idx = i
                break
        row_positions.append((rn, idx))

    def is_x_line(s):
        return norm(s) == "x"

    def has_adjacent_x(slice_lines, idx, keyword_variants, lookahead=1, lookbehind=1):
        ns = norm(slice_lines[idx])
        matched = any(norm(kw) in ns for kw in keyword_variants)
        if not matched:
            return False

        for kw in keyword_variants:
            if f"x {norm(kw)}" in ns or f"x{norm(kw)}" in ns:
                return True

        for k in range(max(0, idx - lookbehind), idx):
            p, _ = parse_prefix_and_text(slice_lines[k])
            if p == "X" or is_x_line(slice_lines[k]):
                return True

        for k in range(idx + 1, min(len(slice_lines), idx + 1 + lookahead)):
            p, _ = parse_prefix_and_text(slice_lines[k])
            if p == "X" or is_x_line(slice_lines[k]):
                return True

        return False

    def extract_temperature(row_slice):
        for line in row_slice:
            txt = line.replace(",", ".")
            m = re.search(r"\b(3[5-9]\.[0-9])\b", txt)
            if m:
                return m.group(1).replace(".", ",")
        return None

    def extract_selected_parasitologie(row_slice):
        selected = []
        for i, line in enumerate(row_slice):
            nl = norm(line)
            if "absence" in nl:
                if has_adjacent_x(row_slice, i, ["absence"]):
                    selected.append("Absence")
            elif "gameto seuls" in nl or "game to seuls" in nl:
                if has_adjacent_x(row_slice, i, ["gaméto seuls", "gameto seuls"]):
                    selected.append("Gaméto seuls")
            elif "trophos" in nl or "rophos" in nl:
                if has_adjacent_x(row_slice, i, ["trophos"]):
                    selected.append("Trophos")
        out = []
        for x in selected:
            if x not in out:
                out.append(x)
        return out

    def extract_selected_densite(row_slice):
        selected = []
        density_variants = [
            ("≤ 100", ["≤ 100", "<=100", "≤100"]),
            ("101-10 000", ["101-10 000", "101-10000"]),
            ("> 10 000", ["> 10 000", ">10000"]),
        ]
        for i, line in enumerate(row_slice):
            for label, variants in density_variants:
                if any(norm(v) in norm(line) for v in variants):
                    if has_adjacent_x(row_slice, i, variants):
                        selected.append(label)
        out = []
        for x in selected:
            if x not in out:
                out.append(x)
        return out

    # Determine "fait" per row: look for X Oui / X marker near row
    def row_fait(row_slice):
        for line in row_slice:
            prefix, content = parse_prefix_and_text(line)
            if prefix == "X" and norm(content) == "oui":
                return "Oui"
            if prefix == "O" and norm(content) == "oui":
                return "Non"
        return None

    for idx, (rn, pos) in enumerate(row_positions):
        row = {
            "row": rn,
            "fait": None,
            "temperature": None,
            "parasitologie": [],
            "densite_parasitaire": []
        }

        if pos is None:
            result["rows"].append(row)
            continue

        next_positions = [p for _, p in row_positions[idx + 1:] if p is not None]
        end_pos = next_positions[0] if next_positions else len(block)

        row_slice = block[pos:end_pos]

        row["parasitologie"] = extract_selected_parasitologie(row_slice)
        row["densite_parasitaire"] = extract_selected_densite(row_slice)
        row["temperature"] = extract_temperature(row_slice)

        fait_explicit = row_fait(row_slice)
        if fait_explicit:
            row["fait"] = fait_explicit
        elif row["parasitologie"] or row["densite_parasitaire"] or row["temperature"]:
            row["fait"] = "Oui"
        else:
            row["fait"] = "Non"

        result["rows"].append(row)

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse CNR page 4 options from OCR txt")
    parser.add_argument("ocr_txt_path")
    parser.add_argument("--page-num", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page4_ocr_text_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_file_lines = load_ocr_txt(ocr_txt_path)
    raw_page_lines = extract_page_block(raw_file_lines, page_num=args.page_num)

    replacements = [
        ("XHospitalisation", "X Hospitalisation"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
        ("O u Nb total Cpés/J", "Ou Nb total Cpés/J"),
    ]
    lines = postprocess_lines(raw_page_lines, replacements=replacements)

    sections = []
    field_results = []

    for spec in page4_specs():
        sec = slice_section(
            lines=lines,
            start_anchors=spec["start_anchors"],
            end_anchors=spec["end_anchors"],
            max_lines=spec["max_lines"],
        )

        if sec is None:
            sections.append({"field": spec["field"], "found": False})
            field_results.append({"field": spec["field"], "found": False, "options": []})
            continue

        sections.append({
            "field": spec["field"],
            "found": True,
            "anchor_line": sec["anchor_line"],
            "start_idx": sec["start_idx"],
            "end_idx": sec["end_idx"],
            "lines": sec["lines"],
        })

        option_results = []
        for canonical, variants in spec["options"]:
            item = parse_option_from_lines(sec["lines"], canonical, variants)
            option_results.append(item)

        # Fix inline "Effet indésirable? X Non"
        if spec["field"] == "Effet indésirable 1ère intention":
            inline_selected = None
            for line in sec["lines"]:
                inline_selected = detect_inline_x_option(line, ["Non", "Mineur", "Grave"])
                if inline_selected:
                    break

            if inline_selected:
                for item in option_results:
                    if item["option"] == inline_selected:
                        item["selected"] = True
                        item["decision_source"] = "ocr_prefix_X_inline"

        option_results = postprocess_single_choice(option_results)
        option_results = apply_elimination_heuristic(option_results, spec.get("single_choice", False))
        selected_options = [x["option"] for x in option_results if x.get("selected")]

        field_results.append({
            "field": spec["field"],
            "found": True,
            "selected_options": selected_options,
            "options": option_results,
        })

    controle_block = parse_controle_parasitologique(lines)

    result = {
        "ocr_txt_path": str(ocr_txt_path),
        "page_num": args.page_num,
        "lines_count": len(lines),
        "ocr_lines_preview": lines[:150],
        "sections": sections,
        "field_results": field_results,
        "custom_blocks": [controle_block],
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_page{args.page_num}_ocr_parsed.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lines loaded:       {len(lines)}")
    print(f"Fields parsed:      {len(field_results)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
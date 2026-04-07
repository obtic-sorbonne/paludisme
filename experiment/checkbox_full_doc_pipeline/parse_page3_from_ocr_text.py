from pathlib import Path
import argparse
import json

from cnr_common import (
    clean_text,
    norm,
    compact_norm,
    parse_prefix_and_text,
    text_matches_option,
    load_ocr_txt,
    extract_page_block,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    find_first_line_index,
    is_lone_marker,
    line_select_state,
)


def page3_specs():
    return [
        {
            "field": "Espèce(s) Plasmodiale(s)",
            "single_choice": False,
            "start_anchors": ["Espèce(s) Plasmodiale(s)"],
            "end_anchors": ["Hémoglobine", "Hemoglobine"],
            "max_lines": 10,
            "options": [
                ("P falciparum", ["P falciparum"]),
                ("P ovale", ["P ovale", "Povale"]),
                ("Plasmodium spp", ["Plasmodium spp"]),
                ("P vivax", ["P vivax", "Pvivax"]),
                ("P malariae", ["P malariae"]),
            ],
        },
        {
            "field": "Bandelettes",
            "single_choice": True,
            "start_anchors": ["Bandelettes", "Bandelettes (HRP2"],
            "end_anchors": ["Autres techniques", "Commentaire"],
            "max_lines": 10,
            "options": [
                ("Fait", ["Fait"]),
                ("Non fait", ["Non fait"]),
                ("positif Ag P falciparum", ["positif Ag P falciparum"]),
                ("positif Ag commun", ["positif Ag commun"]),
                ("Ag autres espèces", ["Ag autres espèces", "Ag autres especes"]),
            ],
        },
        {
            "field": "Autres techniques",
            "single_choice": False,
            "start_anchors": ["Autres techniques"],
            "end_anchors": ["Commentaire", "Lame transmise"],
            "max_lines": 8,
            "options": [
                ("PCR", ["PCR"]),
                ("QBC", ["QBC"]),
                ("Sérologie", ["Sérologie", "Serologie"]),
                ("Autres", ["Autres"]),
            ],
        },
        {
            "field": "Lame transmise par autre Labo",
            "single_choice": True,
            "start_anchors": ["Lame transmise par autre Labo"],
            "end_anchors": ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"],
            "max_lines": 6,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
            ],
        },
        {
            "field": "Protection Personnelle Anti-Moustiques",
            "single_choice": True,
            "start_anchors": ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"],
            "end_anchors": ["Répulsifs cutanés", "Repulsifs cutanes"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Répulsifs cutanés",
            "single_choice": True,
            "start_anchors": ["Répulsifs cutanés", "Repulsifs cutanes"],
            "end_anchors": ["Moustiquaires de lit"],
            "max_lines": 6,
            "options": [
                ("Sans autre indication", ["Sans autre indication"]),
                ("Régulier", ["Régulier", "Regulier"]),
                ("Episodique", ["Episodique"]),
            ],
        },
        {
            "field": "Moustiquaires de lit",
            "single_choice": True,
            "start_anchors": ["Moustiquaires de lit"],
            "end_anchors": ["Autres, préciser", "Autres, preciser", "Chimioprophylaxie"],
            "max_lines": 6,
            "options": [
                ("Sans autre indication", ["Sans autre indication"]),
                ("Régulier", ["Régulier", "Regulier"]),
                ("Episodique", ["Episodique"]),
            ],
        },
        {
            "field": "Chimioprophylaxie utilisée",
            "single_choice": True,
            "start_anchors": ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"],
            "end_anchors": ["Date de la dernière prise", "Date de la demière prise", "Arrêt"],
            "max_lines": 35,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
                ("Atovaquone + Proguanil (Malarone®)", ["Atovaquone + Proguanil", "Malarone"]),
                ("Méfloquine (Lariam®)", ["Méfloquine", "Lariam", "Mefloquine"]),
                ("Doxycycline (Doxypalu®)", ["Doxycycline", "Doxypalu"]),
                ("Chloroquine + Proguanil (ou Savarine®)", ["Chloroquine + Proguanil", "Savarine"]),
                ("Chloroquine (Nivaquine®)", ["Chloroquine (Nivaquine®)", "Nivaquine"]),
                ("Proguanil (Paludrine®)", ["Proguanil (Paludrine®)", "Paludrine"]),
                ("Autre, préciser", ["Autre, préciser", "Autre, preciser"]),
            ],
        },
        {
            "field": "Arrêt de la prise suite à intolérance/effet(s) secondaire(s)",
            "single_choice": True,
            "start_anchors": ["Arrêt de la prise suite à intolérance", "Arret de la prise suite a intolerance"],
            "end_anchors": None,
            "max_lines": 6,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
    ]


def parse_protection_personnelle_block(lines):
    result = {
        "field": "Protection Personnelle Anti-Moustiques",
        "found": False,
        "selected": None,
        "details": []
    }

    start_idx = find_first_line_index(lines, ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"])
    if start_idx is None:
        return result

    result["found"] = True

    block_top = lines[start_idx:start_idx + 8]
    for line in block_top:
        prefix, content = parse_prefix_and_text(line)
        txt = clean_text(content if prefix else line)
        if norm(txt) in {"oui", "non", "nsp"} and prefix == "X":
            result["selected"] = txt

    row_specs = [
        ("Répulsifs cutanés", ["Répulsifs cutanés", "Repulsifs cutanes"]),
        ("Moustiquaires de lit", ["Moustiquaires de lit"]),
        ("Autres, préciser", ["Autres, préciser", "Autres, preciser"]),
    ]

    row_options = ["Sans autre indication", "Régulier", "Episodique"]

    for row_name, anchors in row_specs:
        idx = find_first_line_index(lines, anchors)
        if idx is None:
            continue

        row_slice = lines[idx: idx + 6]
        selected = None
        found_options = []

        for line in row_slice:
            prefix, content = parse_prefix_and_text(line)
            txt = clean_text(content if prefix else line)

            for opt in row_options:
                if text_matches_option(txt, opt):
                    found_options.append(opt)
                    if prefix == "X":
                        selected = opt

        result["details"].append({
            "item": row_name,
            "selection": selected,
            "options_found": found_options
        })

    return result


def parse_chimioprophylaxie_block(lines):
    result = {
        "field": "Chimioprophylaxie utilisée",
        "found": False,
        "selected": None,
        "table_results": []
    }

    start_idx = find_first_line_index(lines, ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"])
    if start_idx is None:
        return result

    result["found"] = True

    heading_slice = lines[start_idx:start_idx + 8]
    for line in heading_slice:
        prefix, content = parse_prefix_and_text(line)
        txt = clean_text(content if prefix else line)
        if norm(txt) in {"oui", "non", "nsp"} and prefix == "X":
            result["selected"] = txt

    medicines = [
        "Atovaquone + Proguanil (Malarone®)",
        "Méfloquine (Lariam®)",
        "Doxycycline (Doxypalu®)",
        "Chloroquine + Proguanil (ou Savarine®)",
        "Chloroquine (Nivaquine®)",
        "Proguanil (Paludrine®)",
        "Autre, préciser",
    ]

    medicine_variants = {
        "Atovaquone + Proguanil (Malarone®)": ["Atovaquone + Proguanil (Malarone®)", "Atovaquone + Proguanil"],
        "Méfloquine (Lariam®)": ["Méfloquine (Lariam®)", "Méfloquine", "Lariam"],
        "Doxycycline (Doxypalu®)": ["Doxycycline (Doxypalu®)", "Doxycycline"],
        "Chloroquine + Proguanil (ou Savarine®)": ["Chloroquine + Proguanil (ou Savarine®)", "Chloroquine + Proguanil", "Savarine"],
        "Chloroquine (Nivaquine®)": ["Chloroquine (Nivaquine®)", "Nivaquine", "Chloroquine"],
        "Proguanil (Paludrine®)": ["Proguanil (Paludrine®)", "Paludrine", "Proguanil"],
        "Autre, préciser": ["Autre, préciser", "Autre, preciser"],
    }

    med_positions = []
    for i, line in enumerate(lines):
        for med in medicines:
            for var in medicine_variants[med]:
                if text_matches_option(line, var):
                    med_positions.append((i, med))
                    break

    dedup = []
    seen = set()
    for idx, med in med_positions:
        key = (idx, med)
        if key not in seen:
            dedup.append((idx, med))
            seen.add(key)
    med_positions = sorted(dedup, key=lambda x: x[0])

    for idx, med in med_positions:
        window_start = max(0, idx - 2)
        window = lines[window_start:idx + 2]
        selected_label = None
        for line in window:
            prefix, content = parse_prefix_and_text(line)
            if prefix == "X":
                if text_matches_option(content, med) or text_matches_option(line, med):
                    selected_label = "Régulier"
                    break

            txt = clean_text(line)
            if txt and txt not in {"X", "O", "•"}:
                if not any(text_matches_option(txt, m) for m in medicines) and not text_matches_option(txt, med):
                    if idx > 0 and is_lone_marker(lines[idx - 1]):
                        mt = is_lone_marker(lines[idx - 1])
                        if mt == "X":
                            selected_label = txt

        if selected_label:
            result["table_results"].append({
                "medicine": med,
                "selection": selected_label
            })

    return result


def parse_bandelettes_block(lines):
    """
    Parse the Bandelettes block as a structured section, not a simple radio row.
    """
    result = {
        "field": "Bandelettes",
        "found": False,
        "status": None,
        "selected_results": [],
        "all_results_found": [],
    }

    start_idx = find_first_line_index(lines, ["Bandelettes", "Bandelettes (HRP2"])
    if start_idx is None:
        return result

    result["found"] = True

    end_idx = None
    for j in range(start_idx + 1, min(len(lines), start_idx + 15)):
        ln = norm(lines[j])
        if "autres techniques" in ln or "commentaire" in ln:
            end_idx = j
            break

    if end_idx is None:
        end_idx = min(len(lines), start_idx + 15)

    block = lines[start_idx:end_idx]

    for line in block:
        ln = norm(line)
        if "non fait" in ln:
            result["status"] = "Non fait"
            break

    if result["status"] is None:
        for line in block:
            ln = norm(line)
            if "fait" in ln:
                result["status"] = "Fait"
                break

    option_specs = [
        ("positif Ag P falciparum", ["positif Ag P falciparum"]),
        ("positif Ag commun", ["positif Ag commun"]),
        ("Ag autres espèces", ["Ag autres espèces", "Ag autres especes"]),
    ]

    found_exact = []
    for canonical, variants in option_specs:
        for line in block:
            ln = clean_text(line)
            if any(text_matches_option(ln, v) for v in variants):
                found_exact.append(canonical)
                break

    dedup = []
    for x in found_exact:
        if x not in dedup:
            dedup.append(x)

    result["all_results_found"] = dedup

    if result["status"] == "Fait" and dedup:
        result["selected_results"] = [dedup[0]]

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse CNR page 3 options from OCR txt")
    parser.add_argument("ocr_txt_path")
    parser.add_argument("--page-num", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page3_ocr_text_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_file_lines = load_ocr_txt(ocr_txt_path)
    raw_page_lines = extract_page_block(raw_file_lines, page_num=args.page_num)

    replacements = [
        ("ONSP", "O NSP"),
        ("Povale", "P ovale"),
        ("Pvivax", "P vivax"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
    ]
    lines = postprocess_lines(raw_page_lines, replacements=replacements)

    sections = []
    field_results = []
    custom_blocks = []

    for spec in page3_specs():
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
            option_results.append(parse_option_from_lines(sec["lines"], canonical, variants))

        option_results = postprocess_single_choice(option_results)
        option_results = apply_elimination_heuristic(option_results, spec.get("single_choice", False))
        selected_options = [x["option"] for x in option_results if x["selected"]]

        field_results.append({
            "field": spec["field"],
            "found": True,
            "selected_options": selected_options,
            "options": option_results,
        })

    custom_blocks.append(parse_protection_personnelle_block(lines))
    custom_blocks.append(parse_chimioprophylaxie_block(lines))
    custom_blocks.append(parse_bandelettes_block(lines))

    result = {
        "ocr_txt_path": str(ocr_txt_path),
        "page_num": args.page_num,
        "lines_count": len(lines),
        "ocr_lines_preview": lines[:150],
        "sections": sections,
        "field_results": field_results,
        "custom_blocks": custom_blocks,
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_page{args.page_num}_ocr_parsed.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lines loaded:       {len(lines)}")
    print(f"Fields parsed:      {len(field_results)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
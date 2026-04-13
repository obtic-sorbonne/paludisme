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
        "rows": [],
        "commentaires_remarques": None,
        "perdu_de_vue": None,
    }

    def n(s):
        return norm(clean_text(s))

    def has_any(txt, variants):
        t = n(txt)
        return any(v in t for v in variants)

    # -----------------------------
    # Find control block start
    # -----------------------------
    start_idx = None
    start_variants = [
        "controle parasitologique p falciparum",
        "contrôle parasitologique p falciparum",
        "parasitologique p falciparum",
        "ontrôle parasitologique p falciparum",
        "ontrole parasitologique p falciparum",
    ]

    for i, line in enumerate(lines):
        if has_any(line, start_variants):
            start_idx = i
            break

    if start_idx is None:
        return result

    result["found"] = True
    block = lines[start_idx:start_idx + 60]

    # -----------------------------
    # Overall Oui / Non
    # -----------------------------
    for i in range(min(6, len(block))):
        txt = clean_text(block[i])
        prefix, content = parse_prefix_and_text(txt)

        if prefix == "X" and n(content) in {"oui", "non"}:
            result["control_overall"] = clean_text(content).capitalize()
            break

        if prefix == "X" and i + 1 < len(block):
            nxt = n(block[i + 1])
            if nxt in {"oui", "non"}:
                result["control_overall"] = clean_text(block[i + 1]).capitalize()
                break

    # -----------------------------
    # Row positions
    # -----------------------------
    row_positions = []
    for rn in row_names:
        idx = None
        rn_norm = n(rn)
        for i, line in enumerate(block):
            if rn_norm in n(line):
                idx = i
                break
        row_positions.append((rn, idx))

    def is_x_line(s):
        return n(s) == "x"

    def has_adjacent_x(slice_lines, idx, keyword_variants, lookahead=1, lookbehind=1):
        ns = n(slice_lines[idx])
        matched = any(n(kw) in ns for kw in keyword_variants)
        if not matched:
            return False

        for kw in keyword_variants:
            nkw = n(kw)
            if f"x {nkw}" in ns or f"x{nkw}" in ns:
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
            txt = clean_text(line).replace(",", ".")
            m = re.search(r"\b(3[5-9]\.[0-9])\b", txt)
            if m:
                return m.group(1).replace(".", ",")
        return None

    def extract_selected_parasitologie(row_slice):
        selected = []
        for i, line in enumerate(row_slice):
            nl = n(line)

            if "absence" in nl:
                if has_adjacent_x(row_slice, i, ["absence"]):
                    selected.append("Absence")
            elif "gameto seuls" in nl or "gaméto seuls" in nl:
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
                if any(n(v) in n(line) for v in variants):
                    if has_adjacent_x(row_slice, i, variants):
                        selected.append(label)

        out = []
        for x in selected:
            if x not in out:
                out.append(x)
        return out

    def row_fait(row_slice):
        for line in row_slice:
            prefix, content = parse_prefix_and_text(line)
            if prefix == "X" and n(content) == "oui":
                return "Oui"
            if prefix == "O" and n(content) == "oui":
                return "Non"
        return None

    # -----------------------------
    # Parse rows
    # -----------------------------
    for idx, (rn, pos) in enumerate(row_positions):
        row = {
            "row": rn,
            "fait": None,
            "temperature": None,
            "parasitologie": [],
            "densite_parasitaire": [],
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
            row["fait"] = None

        result["rows"].append(row)

        # -----------------------------
    # Commentaires & Remarques
    # -----------------------------
    comment_idx = None
    comment_variants = [
        "commentaires",
        "ommentaires",
        "remarques",
    ]

    for i, line in enumerate(block):
        nl = norm(line)
        if any(v in nl for v in comment_variants):
            comment_idx = i
            break

    if comment_idx is not None:
        comment_parts = []

        for j in range(comment_idx, min(len(block), comment_idx + 15)):
            txt = clean_text(block[j])
            nt = norm(txt)

            if not txt:
                continue

            if (
                "perdu de vue" in nt
                or "validation senior" in nt
                or "pages visitees" in nt
                or "pages visitées" in nt
                or "voozanoo" in nt
                or "http" in nt
                or "centre national de référence" in nt
            ):
                break

            # skip the label line itself, but keep text after colon if present
            if "commentaires" in nt or "remarques" in nt:
                if ":" in txt:
                    right = clean_text(txt.split(":", 1)[1])
                    if right:
                        comment_parts.append(right)
                continue

            # skip OCR garbage-only lines
            if txt in {"E", "e", "©"}:
                continue

            comment_parts.append(txt)

        if comment_parts:
            result["commentaires_remarques"] = " ".join(comment_parts)
    
    
    # -----------------------------
    # Perdu de vue
    # -----------------------------
    for i, line in enumerate(block):
        if "perdu de vue" in n(line):
            prefix, content = parse_prefix_and_text(line)
            if prefix == "X":
                result["perdu_de_vue"] = "Oui"
            elif prefix == "O":
                result["perdu_de_vue"] = "Non"
            else:
                result["perdu_de_vue"] = None
            break

    return result


def parse_page4_treatment_block(lines):
    result = {
        "field": "Traitement et hospitalisation",
        "found": False,
        "prise_en_charge": None,
        "date_premiere_prise_structure": None,
        "nombre_de_jours_hospitalisation": None,
        "dont_reanimation_si": None,
        "transfert_autre_hopital": None,
        "poids_kg": None,
        "traitement_antipalustre": [],
        "traitement_debute_le": None,
        "dose_totale_mg_j": None,
        "duree_jours": None,
        "commentaires": None,
    }

    start_idx = None
    for i, line in enumerate(lines):
        if "prise en charge" in norm(line) and "traitement" in norm(line):
            start_idx = i
            break

    if start_idx is None:
        return result

    result["found"] = True
    block = lines[start_idx:start_idx + 60]

    def find_line_index_contains(block_lines, variants):
        for i, line in enumerate(block_lines):
            nl = norm(line)
            for v in variants:
                if norm(v) in nl:
                    return i
        return None

    def normalize_ocr_numeric_text(txt: str) -> str:
        txt = clean_text(txt)
        txt = re.sub(r"\bI1\b", "1", txt)
        txt = re.sub(r"\bl1\b", "1", txt)
        txt = re.sub(r"\bIl\b", "1", txt)
        txt = re.sub(r"\bI\b", "1", txt)
        txt = re.sub(r"\bl\b", "1", txt)
        txt = re.sub(r"\bO\b", "0", txt)
        return txt

    def extract_traitement_debute_le():
        idx = find_line_index_contains(block, ["Traitement débuté le", "Traitement debute le"])
        if idx is None:
            return None

        local = block[idx: idx + 4]

        for line in local:
            txt = clean_text(line)
            ntxt = norm(txt)

            if "soir" in ntxt:
                return "Soir"
            if "matin" in ntxt:
                return "Matin"
            if "midi" in ntxt:
                return "Midi"

            if ":" in txt:
                right = clean_text(txt.split(":", 1)[1])
                if right:
                    return right

        return None

    def extract_first_date_after(anchor_variants, lookahead=6):
        idx = find_line_index_contains(block, anchor_variants)
        if idx is None:
            return None

        local = block[idx:idx + lookahead + 1]

        def normalize_ocr_date_text(txt: str) -> str:
            txt = clean_text(txt)
            txt = txt.replace("2c", "2006")
            txt = txt.replace("2C", "2006")
            txt = txt.replace("2o06", "2006")
            txt = txt.replace("20o6", "2006")
            txt = txt.replace("2O06", "2006")
            return txt

        for line in local:
            txt = normalize_ocr_date_text(line)
            m = re.search(r"\b(\d{2})[^\d]{0,3}(\d{2})[^\d]{0,3}(\d{4})\b", txt)
            if m:
                dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
                try:
                    if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                        return f"{dd}/{mm}/{yyyy}"
                except Exception:
                    pass

        nums = []
        for line in local:
            txt = normalize_ocr_date_text(line)
            if "(jj/mm" in norm(txt):
                continue
            found = re.findall(r"\b\d{1,4}\b", txt)
            nums.extend(found)

        for k in range(len(nums) - 2):
            a, b, c = nums[k], nums[k + 1], nums[k + 2]
            if len(a) <= 2 and len(b) <= 2 and len(c) == 4:
                dd = a.zfill(2)
                mm = b.zfill(2)
                yyyy = c
                try:
                    if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                        return f"{dd}/{mm}/{yyyy}"
                except Exception:
                    pass

        return None

    def extract_first_integer_after(anchor_variants, lookahead=4):
        idx = find_line_index_contains(block, anchor_variants)
        if idx is None:
            return None

        local = block[idx:min(len(block), idx + lookahead + 1)]

        for txt in local:
            txt = normalize_ocr_numeric_text(txt)
            nums = re.findall(r"\b\d+\b", txt)
            if nums:
                return nums[0]

        return None

    def has_explicit_x_for_option(option_variants):
        for i, line in enumerate(block):
            raw = clean_text(line)
            nline = norm(raw)

            for opt in option_variants:
                nopt = norm(opt)

                if f"x {nopt}" in nline or nline.startswith(f"x {nopt}"):
                    return True

                prefix, content = parse_prefix_and_text(raw)
                if prefix == "X" and nopt in norm(content):
                    return True

                if nopt in nline:
                    for j in range(max(0, i - 1), min(len(block), i + 2)):
                        pfx, _ = parse_prefix_and_text(block[j])
                        if pfx == "X":
                            return True
                        if norm(clean_text(block[j])) == "x":
                            return True

        return False

    def has_explicit_o_for_option(option_variants):
        for raw in block:
            nline = norm(raw)
            for opt in option_variants:
                nopt = norm(opt)
                if f"o {nopt}" in nline or nline.startswith(f"o {nopt}"):
                    return True
                prefix, content = parse_prefix_and_text(raw)
                if prefix == "O" and nopt in norm(content):
                    return True
        return False

    result["traitement_debute_le"] = extract_traitement_debute_le()

    result["date_premiere_prise_structure"] = extract_first_date_after(
        ["Date de la première prise médicamenteuse dans votre structure de soin"]
    )

    prise = []
    if has_explicit_x_for_option(["Ambulatoire"]):
        prise.append("Ambulatoire")
    if has_explicit_x_for_option(["Hospitalisation"]):
        prise.append("Hospitalisation")

    if prise:
        result["prise_en_charge"] = ", ".join(prise)

    result["nombre_de_jours_hospitalisation"] = extract_first_integer_after(
        ["Nombre de jours d’hospitalisation", "Nombre de jours d'hospitalisation", "Nombre de jours d’hopistalisation"]
    )

    result["dont_reanimation_si"] = extract_first_integer_after(
        ["dont réanimation/SI", "dont reanimation/SI"],
        lookahead=3,
    )

    if result["dont_reanimation_si"] is None:
        rea_idx = find_line_index_contains(block, ["dont réanimation/SI", "dont reanimation/SI"])
        if rea_idx is not None:
            local = block[rea_idx:min(len(block), rea_idx + 3)]
            if any(norm(x) in {"o", "0"} for x in local):
                result["dont_reanimation_si"] = "0"

    if has_explicit_x_for_option(["Transfert autre hôpital", "Transfert autre hopital"]):
        result["transfert_autre_hopital"] = "Oui"
    elif has_explicit_o_for_option(["Transfert autre hôpital", "Transfert autre hopital"]):
        result["transfert_autre_hopital"] = "Non"

    result["poids_kg"] = extract_first_integer_after(["Poids", "Poids (Kgs)"], lookahead=3)

    meds = [
        "Halofantrine",
        "Quinine",
        "Riamet",
        "Malarone",
        "Artéméther",
        "Artemether",
        "Artesunate",
        "Nivaquine",
        "Doxycycline",
        "Clindamycine",
    ]

    selected_meds = []
    for med in meds:
        if has_explicit_x_for_option([med]):
            selected_meds.append(med)

    if not selected_meds:
        treat_idx = find_line_index_contains(
            block,
            ["Traitement anti-palustre de 1ère", "Traitement anti-palustre", "Traitement antipalustre"]
        )
        if treat_idx is not None:
            local = block[treat_idx:min(len(block), treat_idx + 8)]

            found_meds = []
            for med in meds:
                for line in local:
                    if norm(med) in norm(line):
                        found_meds.append(med)
                        break

            if len(found_meds) == 1:
                selected_meds = found_meds

    result["traitement_antipalustre"] = selected_meds

    result["dose_totale_mg_j"] = extract_first_nonzero_integer_after(
        ["Dose totale (mg/J)", "Dose totale"],
        lookahead=4,
    )

    result["duree_jours"] = extract_first_integer_after(
        ["Durée en jours", "Duree en jours"],
        lookahead=3,
    )

    comment_idx = find_line_index_contains(block, ["Commentaire", "Commentaires"])
    if comment_idx is not None:
        comment_parts = []
        for j in range(comment_idx, min(len(block), comment_idx + 6)):
            txt = clean_text(block[j])
            if not txt:
                continue
            if "controle parasitologique" in norm(txt):
                break
            if ":" in txt and j == comment_idx:
                right = clean_text(txt.split(":", 1)[1])
                if right:
                    comment_parts.append(right)
            elif j > comment_idx:
                comment_parts.append(txt)

        if comment_parts:
            result["commentaires"] = " ".join(comment_parts)

    if not result["prise_en_charge"]:
        prise_idx = find_line_index_contains(block, ["Ambulatoire", "Hospitalisation"])
        if prise_idx is not None:
            local = block[max(0, prise_idx - 2): prise_idx + 4]

            has_ambulatoire_o = any("o ambulatoire" in norm(x) for x in local)
            has_hosp_o = any("o hospitalisation" in norm(x) for x in local)

            if not has_hosp_o and any("hospitalisation" in norm(x) for x in local):
                result["prise_en_charge"] = "Hospitalisation"
            elif not has_ambulatoire_o and any("ambulatoire" in norm(x) for x in local):
                result["prise_en_charge"] = "Ambulatoire"

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
    treatment_block = parse_page4_treatment_block(lines)

    result = {
        "ocr_txt_path": str(ocr_txt_path),
        "page_num": args.page_num,
        "lines_count": len(lines),
        "ocr_lines_preview": lines[:150],
        "sections": sections,
        "field_results": field_results,
        "custom_blocks": [controle_block, treatment_block],
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_page{args.page_num}_ocr_parsed.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lines loaded:       {len(lines)}")
    print(f"Fields parsed:      {len(field_results)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
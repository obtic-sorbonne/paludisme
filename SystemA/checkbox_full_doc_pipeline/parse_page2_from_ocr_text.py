from pathlib import Path
import argparse
import json
import re

from cnr_common import (
    load_ocr_txt, extract_page_block, postprocess_lines,
    slice_section, parse_option_from_lines, postprocess_single_choice,
    apply_elimination_heuristic,
    norm, clean_text,
)
from cnr_common import (
    clean_text,
    norm,
    parse_prefix_and_text,
    text_matches_option,
)

def page2_specs():
    return [
        {
            "field": "Fréquence des séjours",
            "single_choice": True,
            # Options for this field bleed from page 1 header onto page 2
            "start_anchors": ["1 ou moins", "Fréquence des séjours", "Frequence des sejours"],
            "end_anchors": ["Date des Premiers Symptomes", "Date des premiers symptômes", "Patient adressé"],
            "max_lines": 8,
            "options": [
                ("1 ou moins", ["1 ou moins"]),
                ("> 1", ["> 1"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Patient adressé",
            "single_choice": True,
            "start_anchors": ["Patient adressé par un", "Patient adresse par un", "Patient adressé"],
            "end_anchors": ["consultation médicale avant", "Y a-t-il eu"],
            "max_lines": 10,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Consultation avant",
            "single_choice": True,
            "start_anchors": ["consultation médicale avant", "Y a-t-il eu une consultation"],
            "end_anchors": ["Etat clinique", "Date de la consultation"],
            "max_lines": 10,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Etat clinique au moment du diagnostic",
            "single_choice": True,
            "start_anchors": ["Etat clinique au moment du diagnostic"],
            "end_anchors": ["Antécédents de paludisme", "Antecedents de paludisme"],
            "max_lines": 12,
            "options": [
                ("Accès simple sans vomissements", ["Accès simple sans vomissements"]),
                ("Accès simple AVEC vomissements", ["Accès simple AVEC vomissements"]),
                ("Formes Asymptomatiques et découvertes fortuites", ["Formes Asymptomatiques et découvertes fortuites"]),
                ("Accès GRAVE", ["Accès GRAVE"]),
                ("Paludisme Viscéral évolutif (PVE)", ["Paludisme Viscéral évolutif"]),
            ],
        },
        {
            "field": "Antécédents de paludisme dans les 3 derniers mois",
            "single_choice": True,
            "start_anchors": ["Antécédents de paludisme dans les 3 derniers mois"],
            "end_anchors": ["S'agit-il d'une femme enceinte ou parturiente"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Femme enceinte ou parturiente",
            "single_choice": True,
            "start_anchors": ["S'agit-il d'une femme enceinte ou parturiente"],
            "end_anchors": ["S'agit-il d'un patient ayant une immunodépression"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Immunodépression connue",
            "single_choice": True,
            "start_anchors": ["S'agit-il d'un patient ayant une immunodépression"],
            "end_anchors": ["Paludismes \"autochtones\"", "Paludismes autochtones"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Dialyse",
            "single_choice": True,
            "start_anchors": ["Dialyse"],
            "end_anchors": ["Si glycémie valeur glycémie", "Si autres critères de gravité"],
            "max_lines": 4,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
            ],
        },
        {
            "field": "Paludismes autochtones",
            "single_choice": True,
            "start_anchors": ["Paludismes \"autochtones\"", "Paludismes autochtones"],
            "end_anchors": None,
            "max_lines": 12,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("Congénital", ["Congénital"]),
                ("Accidentel", ["Accidentel"]),
                ("Transfusionnel", ["Transfusionnel"]),
                ("Aéroportuaire", ["Aéroportuaire"]),
                ("Suspicion d'autochtone vrai", ["Suspicion d'autochtone vrai"]),
                ("Cryptique", ["Cryptique"]),
            ],
        },
    ]



def parse_page2_clinical_block(lines):
    """
    Grouped page-2 clinical block for cleaner final output.

    Includes:
      - Date de la consultation actuelle
      - Etat clinique au moment du diagnostic
      - Antécédents de paludisme dans les 3 derniers mois
      - Femme enceinte ou parturiente
      - Immunodépression connue
      - Paludismes autochtones
    """

    result = {
        "field": "Contexte clinique page 2",
        "found": False,
        "date_consultation_actuelle": None,
        "etat_clinique": None,
        "antecedents_paludisme_3m": None,
        "femme_enceinte_ou_parturiente": None,
        "immunodepression_connue": None,
        "paludismes_autochtones": None,
    }

    def _n(s):
        return norm(clean_text(s))

    def _find_anchor_idx(anchor_variants):
        for i, line in enumerate(lines):
            ln = _n(line)
            for a in anchor_variants:
                if _n(a) in ln:
                    return i
        return None

    def _extract_date_near(anchor_variants, lookahead=6):
        idx = _find_anchor_idx(anchor_variants)
        if idx is None:
            return None

        local = lines[idx: min(len(lines), idx + lookahead + 1)]

        for line in local:
            txt = clean_text(line)
            m = re.search(r"\b(\d{2})[^\d]{0,3}(\d{2})[^\d]{0,3}(\d{4})\b", txt)
            if m:
                dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
                try:
                    if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                        return f"{dd}/{mm}/{yyyy}"
                except Exception:
                    pass
        return None

    def _extract_selected_option_near(anchor_variants, option_variants_map, lookahead=8):
        idx = _find_anchor_idx(anchor_variants)
        if idx is None:
            return None

        local = lines[idx: min(len(lines), idx + lookahead + 1)]

        for line in local:
            prefix, content = parse_prefix_and_text(line)
            txt = clean_text(content if prefix else line)

            for canonical, variants in option_variants_map:
                if any(text_matches_option(txt, v) for v in variants):
                    if prefix == "X":
                        return canonical

        # fallback for circle-dot OCR style where selected option may appear as plain visible text
        found_plain = []
        for line in local:
            txt = clean_text(line)
            for canonical, variants in option_variants_map:
                if any(text_matches_option(txt, v) for v in variants):
                    if canonical not in found_plain:
                        found_plain.append(canonical)

        if len(found_plain) == 1:
            return found_plain[0]

        return None

    result["date_consultation_actuelle"] = _extract_date_near(
        ["Date de la consultation actuelle"]
    )

    result["etat_clinique"] = _extract_selected_option_near(
        ["Etat clinique au moment du diagnostic", "État clinique au moment du diagnostic"],
        [
            ("Accès simple sans vomissements", ["Accès simple sans vomissements", "Acces simple sans vomissements"]),
            ("Accès simple AVEC vomissements", ["Accès simple AVEC vomissements", "Acces simple AVEC vomissements"]),
            ("Formes Asymptomatiques et découvertes fortuites", ["Formes Asymptomatiques et découvertes fortuites"]),
            ("Accès GRAVE", ["Accès GRAVE", "Acces GRAVE"]),
            ("Paludisme Viscéral évolutif (PVE)", ["Paludisme Viscéral évolutif", "Paludisme Visceral evolutif", "PVE"]),
        ],
        lookahead=12,
    )

    result["antecedents_paludisme_3m"] = _extract_selected_option_near(
        ["Antécédents de paludisme", "Antecedents de paludisme"],
        [
            ("Oui", ["Oui"]),
            ("Non", ["Non"]),
            ("NSP", ["NSP"]),
        ],
        lookahead=4,
    )

    result["femme_enceinte_ou_parturiente"] = _extract_selected_option_near(
        ["femme enceinte", "parturiente"],
        [
            ("Oui", ["Oui"]),
            ("Non", ["Non"]),
            ("NSP", ["NSP"]),
        ],
        lookahead=4,
    )

    result["immunodepression_connue"] = _extract_selected_option_near(
        ["immunodépression", "immunodepression"],
        [
            ("Oui", ["Oui"]),
            ("Non", ["Non"]),
            ("NSP", ["NSP"]),
        ],
        lookahead=4,
    )

    result["paludismes_autochtones"] = _extract_selected_option_near(
        ["Paludismes autochtones", 'Paludismes "autochtones"'],
        [
            ("Oui", ["Oui"]),
            ("Non", ["Non"]),
            ("NSP", ["NSP"]),
        ],
        lookahead=4,
    )

    result["found"] = any(
        [
            result["date_consultation_actuelle"],
            result["etat_clinique"],
            result["antecedents_paludisme_3m"],
            result["femme_enceinte_ou_parturiente"],
            result["immunodepression_connue"],
            result["paludismes_autochtones"],
        ]
    )

    return result



def main():
    parser = argparse.ArgumentParser(description="Parse CNR page 2 options from OCR txt")
    parser.add_argument("ocr_txt_path")
    parser.add_argument("--page-num", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page2_ocr_text_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_file_lines = load_ocr_txt(ocr_txt_path)
    raw_page_lines = extract_page_block(raw_file_lines, page_num=args.page_num)

    replacements = [
        ("OAccès", "O Accès"),
        ("OMilitaires", "O Militaires"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
    ]
    lines = postprocess_lines(raw_page_lines, replacements=replacements)

    sections = []
    field_results = []

    for spec in page2_specs():
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

    result = {
        "ocr_txt_path": str(ocr_txt_path),
        "page_num": args.page_num,
        "lines_count": len(lines),
        "ocr_lines_preview": lines[:150],
        "sections": sections,
        "field_results": field_results,
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_page{args.page_num}_ocr_parsed.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lines loaded:       {len(lines)}")
    print(f"Fields parsed:      {len(field_results)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
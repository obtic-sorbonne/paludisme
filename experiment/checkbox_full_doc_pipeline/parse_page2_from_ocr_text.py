from pathlib import Path
import argparse
import json

from cnr_common import (
    load_ocr_txt, extract_page_block, postprocess_lines,
    slice_section, parse_option_from_lines, postprocess_single_choice,
    apply_elimination_heuristic,
    norm, clean_text,
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
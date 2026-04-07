from pathlib import Path
import argparse
import json
import re

from cnr_common import (
    load_ocr_txt,
    clean_text,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    extract_page_block,
    norm,
    compact_norm,
    parse_prefix_and_text,
    line_select_state,
    text_matches_option,
    is_lone_marker,
)


# --------------------------------------------------
# Page 1 config
# --------------------------------------------------

def page1_specs():
    return [
        {
            "field": "Sexe",
            "single_choice": True,
            "start_anchors": ["Sexe"],
            "end_anchors": ["Ethnicité", "Ethnicite"],
            "max_lines": 10,
            "options": [
                ("M", ["M"]),
                ("F", ["F"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Ethnicité",
            "single_choice": True,
            "start_anchors": ["Ethnicité", "Ethnicite"],
            "end_anchors": ["Pays de naissance", "Localisation"],
            "max_lines": 15,
            "options": [
                ("Caucasien", ["Caucasien"]),
                ("Asiatique", ["Asiatique"]),
                ("Africain", ["Africain"]),
                ("Africain vivant en France", ["Africain vivant en France"]),
                ("Africain vivant en Afrique", ["Africain vivant en Afrique"]),
                ("Autre", ["Autre"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Durée du séjour",
            "single_choice": True,
            "start_anchors": ["Durée du séjour", "Duree du sejour"],
            "end_anchors": ["Autres pays d'endémie", "Autres pays d'endemie"],
            "max_lines": 15,
            "options": [
                ("1 semaine ou moins", ["1 semaine ou moins", "1 sem ou moins"]),
                ("2 semaines", ["2 semaines", "2 sem"]),
                ("3 semaines", ["3 semaines", "3 sem"]),
                ("4 semaines", ["4 semaines", "4 sem"]),
                ("1 à 3 mois", ["1 à 3 mois", "1 a 3 mois"]),
                ("> 3 mois", ["> 3 mois"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Autres pays d'endémie",
            "single_choice": True,
            "start_anchors": ["Autres pays d'endémie", "Autres pays d'endemie"],
            "end_anchors": ["Nature du séjour", "Nature du sejour"],
            "max_lines": 10,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Nature du séjour",
            "single_choice": False,
            "start_anchors": ["Nature du séjour", "Nature du sejour"],
            "end_anchors": ["Résidence durant le séjour", "Residence durant le sejour"],
            "max_lines": 18,
            "options": [
                ("Tourisme", ["Tourisme"]),
                ("Affaires / Professionnels", ["Affaires / Professionnels", "Affaires/Professionnels"]),
                ("Migrants en visite au Pays d'origine", ["Migrants en visite au Pays d'origine"]),
                ("Navigants, Marins", ["Navigants, Marins", "Navigants,Marins"]),
                ("Résident ou expatriés ≥ 6 mois", ["Résident ou expatriés >= 6 mois", "Résident ou expatriés 6 mois", "Résident ou expatriés"]),
                ("Routard et/ou conditions de séjour précaires", ["Routard et/ou conditions de séjour précaires"]),
                ("Militaires", ["Militaires"]),
                ("NSP", ["NSP"]),
                ("Autres", ["Autres"]),
            ],
        },
        {
            "field": "Résidence en zone d'endémie",
            "single_choice": True,
            "start_anchors": ["Résidence durant le séjour", "Residence durant le sejour"],
            "end_anchors": ["Fréquence des séjours", "Frequence des sejours"],
            "max_lines": 10,
            "options": [
                ("Urbain strict", ["Urbain strict"]),
                ("Rural", ["Rural"]),
                ("Itinérant / Mixte", ["Itinérant / Mixte", "Itinerant / Mixte", "Itinérant/Mixte"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Fréquence des séjours",
            "single_choice": True,
            # This field header is on page 1 but options often bleed to page 2.
            # We extend max_lines and accept options from whichever page finds them.
            "start_anchors": ["Fréquence des séjours", "Frequence des sejours"],
            "end_anchors": ["Date des Premiers Symptomes", "Date des premiers symptômes", "Patient adressé"],
            "max_lines": 15,
            "options": [
                ("1 ou moins", ["1 ou moins"]),
                ("> 1", ["> 1"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Patient adressé",
            "single_choice": True,
            "start_anchors": ["Patient adressé par un", "Patient adresse par un"],
            "end_anchors": ["consultation médicale avant", "consultation medicale avant", "Y a-t-il eu"],
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
            "end_anchors": ["recherche biologique de Paludisme", "Etat clinique", "Date de la consultation"],
            "max_lines": 12,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Recherche biologique",
            "single_choice": True,
            "start_anchors": ["recherche biologique de Paludisme", "recherche biologique de paludisme"],
            "end_anchors": ["Si OUI, résultat", "Si OUI, resultat"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
            ],
        },
        {
            "field": "Résultat recherche",
            "single_choice": True,
            "start_anchors": [
                "Si OUI, résultat",
                "Si OUI, resultat",
                "résultat :",
                "resultat :",
            ],
            "end_anchors": None,
            "max_lines": 8,
            "options": [
                ("Positif", ["Positif"]),
                ("Négatif", ["Négatif", "Negatif"]),
                ("Résultat non connu", ["Résultat non connu", "Resultat non connu"]),
            ],
        },
    ]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse CNR page 1 options directly from OCR txt")
    parser.add_argument("ocr_txt_path", help="Path to OCR txt file")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page1_ocr_text_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_file_lines = load_ocr_txt(ocr_txt_path)
    raw_page_lines = extract_page_block(raw_file_lines, page_num=args.page_num)
    lines = postprocess_lines(raw_page_lines)

    specs = page1_specs()
    sections = []
    field_results = []

    for spec in specs:
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
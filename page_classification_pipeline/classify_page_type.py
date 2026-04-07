from pathlib import Path
import argparse
import json
import re
from collections import Counter


def clean_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def norm(s: str) -> str:
    return clean_text(
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
        .replace("’", "'")
    )


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(data):
    return data.get("res", data) if isinstance(data, dict) else data


def extract_texts_from_paddle_json(data):
    texts = []
    root = get_root(data)

    if isinstance(root, dict):
        if "overall_ocr_res" in root and isinstance(root["overall_ocr_res"], dict):
            ocr = root["overall_ocr_res"]
            rec_texts = ocr.get("rec_texts", [])
            if isinstance(rec_texts, list):
                for t in rec_texts:
                    if isinstance(t, str):
                        t = clean_text(t)
                        if t:
                            texts.append(t)

        for key in ["results", "res", "ocr"]:
            if key in root and isinstance(root[key], list):
                for item in root[key]:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        t = clean_text(item["text"])
                        if t:
                            texts.append(t)

    elif isinstance(root, list):
        for item in root:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                t = clean_text(item["text"])
                if t:
                    texts.append(t)
            elif (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 1
                and isinstance(item[1][0], str)
            ):
                t = clean_text(item[1][0])
                if t:
                    texts.append(t)

    return texts


def detect_layout_flags(data):
    root = get_root(data)
    has_table_layout = False
    has_image_layout = False

    if isinstance(root, dict):
        layout = root.get("layout_det_res", {})
        boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            label = clean_text(box.get("label", "")).lower()
            if label == "table":
                has_table_layout = True
            if label == "image":
                has_image_layout = True

    return {
        "has_table_layout": has_table_layout,
        "has_image_layout": has_image_layout,
    }


def count_matches(joined_norm: str, keywords):
    return sum(1 for kw in keywords if kw in joined_norm)


def compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def count_matches_compact(joined_compact: str, keywords):
    return sum(1 for kw in keywords if compact(kw) in joined_compact)


def count_word_matches(joined_norm: str, keywords):
    total = 0
    for kw in keywords:
        pattern = r"\b" + re.escape(norm(kw)) + r"\b"
        if re.search(pattern, joined_norm):
            total += 1
    return total


def classify_page(texts, layout_flags):
    joined = "\n".join(texts)
    joined_norm = norm(joined)
    joined_compact = compact(joined)

    lab_keywords = [
        "description",
        "resultat",
        "unite",
        "valeurs normales",
        "val.",
        "erythrocytes",
        "hemoglobine",
        "hematocrite",
        "leucocytes",
        "plaquettes",
        "cytologie",
        "examens d'hematologie",
        "examens d'hematologie poste",
        "frottis mince",
        "goutte epaisse",
        "parasitemie",
        "rech parasites sang",
        "biochimie generale",
        "examens de sang",
        "temperature",
        "parasitologie",
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "proteines",
        "uree",
        "creatinine",
        "bilirubine",
        "asat",
        "alat",
        "gamma gt",
        "reticulocytes",
        "metamyelocytes",
        "poly neutrophiles",
        "lymphocytes",
        "monocytes",
        "phosphatases alcalines",
        "bilirubine totale",
        "bilirubine conjuguee",
        "pre albumine",
    ]

    form_keywords = [
        "xoui",
        "xnon",
        "x oui",
        "x non",
        "traitement debute le",
        "prise en charge",
        "ambulatoire",
        "hospitalisation",
        "effet indesirable",
        "medicament de 2nde intention",
        "duree du sejour",
        "date retour",
        "date depart",
        "etat clinique au moment du diagnostic",
        "s'agit-il",
        "mettre une croix",
        "paludismes autochtones",
        "quel pays",
        "pays d'origine",
        "notion de voyage recent",
        "sexe",
        "ethnicite",
        "protection personnelle",
        "protection personnelle anti",
        "moustiquaires de lit",
        "repulsifs cutanes",
        "chimioprophylaxie",
        "lame transmise par autre labo",
        "controle parasitologique",
        "femme enceinte",
        "immunodepression",
    ]

    report_keywords = [
        "cru",
        "compte rendu",
        "compte rendu des urgences",
        "compte rendu d'hospitalisation",
        "compte rendu de sejour",
        "motif de la consultation",
        "parametres a l'arrivee",
        "action iao",
        "actions iao",
        "nom et fonction du medecin",
        "motif medical",
        "antecedents",
        "histoire de la maladie",
        "examen clinique",
        "prescriptions",
        "examens complementaires",
        "conclusion medicale",
        "conclusion medicale definitive",
        "modalites de sorties",
        "modalites de sortie",
        "diagnostic principal",
        "diagnostics associes",
        "documents remis au patient",
        "ordonnance(s) de sortie",
        "service des urgences",
        "service des urgences pediatriques",
        "interne(s)",
        "hopital robert debre",
        "hospital robert debre",
        "pediatrie generale",
        "motif d'hospitalisation",
        "dossier patient d'urgence",
        "sortie d'hospitalisation",
        "bonne evolution",
        "bien a vous",
        "chers parents",
        "conclusion",
        "neuropaludisme",
        "evolution et traitement dans le service",
        "traitement de sortie",
    ]

    lab_keywords_compact = [
        "examensdhematologie",
        "frottismince",
        "rechparasitessang",
        "biochimiegenerale",
        "examensdesang",
        "valeursnormales",
        "temperature",
        "parasitologie",
        "reticulocytes",
        "metamyelocytes",
        "polyneutrophiles",
        "lymphocytes",
        "monocytes",
        "sodium",
        "potassium",
        "bicarbonates",
        "creatinine",
        "bilirubine",
        "phosphatasesalcalines",
        "bilirubinetotale",
        "bilirubineconjuguee",
        "prealbumine",
    ]

    report_keywords_compact = [
        "compterendu",
        "compterendudeshospitalisation",
        "compterendudhospitalisation",
        "motifdelaconsultation",
        "parametresalarivee",
        "actionsiao",
        "motifmedical",
        "antecedents",
        "histoiredelamaladie",
        "examenclinique",
        "prescriptions",
        "examenscomplementaires",
        "conclusionmedicale",
        "modalitesdesortie",
        "diagnosticprincipal",
        "diagnosticsassocies",
        "documentsremisaupatient",
        "ordonnancedesortie",
        "service_des_urgences",
        "servicedesurgences",
        "dossierpatientdurgence",
        "sortiedhospitalisation",
        "neuropaludisme",
        "evolutionettraitementdansleservice",
        "traitementdesortie",
    ]

    form_keywords_compact = [
        "traitementdebutele",
        "effetindesirable",
        "dureedusejour",
        "dateretour",
        "datedepart",
        "etatcliniqueaumomentdudiagnostic",
        "mettreunecroix",
        "quelpays",
        "paysdorigine",
        "notiondevoyagerecent",
        "protectionpersonnelle",
        "moustiquairesdelit",
        "repulsifscutanes",
        "chimioprophylaxie",
        "lametransmiseparautrelabo",
        "controleparasitologique",
        "femmeenceinte",
        "immunodepression",
    ]

    cnr_form_markers = [
        "sexe",
        "ethnicite",
        "duree du sejour",
        "date retour",
        "date depart",
        "autres pays d'endemie",
        "nature du sejour",
        "residence durant le sejour en zone d'endemie",
        "frequence des sejours",
        "patient adresse",
        "etat clinique au moment du diagnostic",
        "paludismes autochtones",
        "espece(s) plasmodiale(s)",
        "lame transmise par autre labo",
        "protection personnelle anti",
        "repulsifs cutanes",
        "moustiquaires de lit",
        "chimioprophylaxie utilisee",
        "arret de la prise suite a intolerance",
        "prise en charge",
        "evolution clinique",
        "controle parasitologique p falciparum",
    ]

    cnr_form_markers_compact = [
        "ethnicite",
        "dureedusejour",
        "dateretour",
        "datedepart",
        "autrespaysdendemie",
        "naturedusejour",
        "residencedurantlesejourenzonedendemie",
        "frequencedessejours",
        "etatcliniqueaumomentdudiagnostic",
        "paludismesautochtones",
        "especesplasmodiales",
        "lametransmiseparautrelabo",
        "protectionpersonnelleanti",
        "repulsifscutanes",
        "moustiquairesdelit",
        "chimioprophylaxieutilisee",
        "priseencharge",
        "evolutionclinique",
        "controleparasitologiquepfalciparum",
    ]

    strong_embedded_report_lab_markers = [
        "examen clinique",
        "examens complementaires",
        "conclusion",
        "motif medical",
        "histoire de la maladie",
        "sortie d'hospitalisation",
        "dossier patient d'urgence",
        "compte rendu",
        "hospitalisation",
        "bonne evolution",
        "neuropaludisme",
        "evolution et traitement dans le service",
        "traitement de sortie",
    ]

    mixed_clinical_with_tables_keywords = [
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "proteines",
        "uree",
        "creatinine",
        "bilan hepatique",
        "phosphatases alcalines",
        "bilirubine totale",
        "bilirubine conjuguee",
        "asat",
        "alat",
        "gamma gt",
        "pre albumine",
        "evolution et traitement dans le service",
        "traitement de sortie",
        "conclusion",
    ]

    mixed_clinical_with_tables_keywords_compact = [
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "proteines",
        "uree",
        "creatinine",
        "bilanhepatique",
        "phosphatasesalcalines",
        "bilirubinetotale",
        "bilirubineconjuguee",
        "asat",
        "alat",
        "gammagt",
        "prealbumine",
        "evolutionettraitementdansleservice",
        "traitementdesortie",
        "conclusion",
    ]

    scores = Counter()

    lab_hits = count_matches(joined_norm, lab_keywords) + count_matches_compact(joined_compact, lab_keywords_compact)
    report_hits = count_matches(joined_norm, report_keywords) + count_matches_compact(joined_compact, report_keywords_compact)
    form_hits = count_matches(joined_norm, form_keywords) + count_matches_compact(joined_compact, form_keywords_compact)
    cnr_form_hits = count_matches(joined_norm, cnr_form_markers) + count_matches_compact(joined_compact, cnr_form_markers_compact)
    short_form_hits = count_word_matches(joined_norm, ["oui", "non", "nsp"])
    embedded_report_lab_hits = count_matches(joined_norm, strong_embedded_report_lab_markers)
    mixed_clinical_hits = count_matches(joined_norm, mixed_clinical_with_tables_keywords) + count_matches_compact(joined_compact, mixed_clinical_with_tables_keywords_compact)

    strong_table_headers = all(k in joined_norm for k in ["description", "resultat", "unite"])
    strong_lab_analytes = count_matches(joined_norm, [
        "hemoglobine",
        "hematocrite",
        "leucocytes",
        "plaquettes",
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "uree",
        "creatinine",
        "bilirubine",
        "asat",
        "alat",
        "gamma gt",
        "phosphatases alcalines",
        "reticulocytes",
    ]) >= 3

    scores["lab_table_page"] += lab_hits
    scores["form_page"] += form_hits + short_form_hits
    scores["clinical_report_page"] += report_hits

    if strong_table_headers:
        scores["lab_table_page"] += 4

    if short_form_hits >= 2:
        scores["form_page"] += 2

    if "compte rendu" in joined_norm or "cru" in joined_norm or "compterendu" in joined_compact:
        scores["clinical_report_page"] += 4

    if (
        count_matches(joined_norm, [
            "prescriptions",
            "evolution",
            "conclusion medicale",
            "diagnostic principal",
            "modalites de sorties",
            "modalites de sortie",
            "histoire de la maladie",
            "motif d'hospitalisation",
            "antecedents",
            "examen clinique",
            "examens complementaires",
        ])
        + count_matches_compact(joined_compact, [
            "prescriptions",
            "conclusionmedicale",
            "diagnosticprincipal",
            "modalitesdesortie",
            "histoiredelamaladie",
            "examenclinique",
            "examenscomplementaires",
        ])
    ) >= 2:
        scores["clinical_report_page"] += 4

    if layout_flags["has_table_layout"] and report_hits >= 2:
        scores["clinical_report_page"] += 1

    if layout_flags["has_table_layout"] and lab_hits >= 2:
        scores["lab_table_page"] += 2

    if cnr_form_hits >= 2:
        scores["form_page"] += 5

    if cnr_form_hits >= 4:
        scores["form_page"] += 6
        scores["lab_table_page"] -= 2

    if (
        "controle parasitologique" in joined_norm
        or "controleparasitologique" in joined_compact
        or "chimioprophylaxie" in joined_norm
        or "protection personnelle" in joined_norm
    ):
        scores["form_page"] += 4
        scores["lab_table_page"] -= 1

    if strong_table_headers:
        scores["lab_table_page"] += 5

    if cnr_form_hits >= 2 and lab_hits >= 2:
        scores["form_page"] += 3
        scores["lab_table_page"] -= 2

    if report_hits >= 3 and lab_hits >= 3:
        scores["clinical_report_page"] += 8
        scores["lab_table_page"] -= 3

    if embedded_report_lab_hits >= 2 and lab_hits >= 2:
        scores["clinical_report_page"] += 8
        scores["lab_table_page"] -= 4

    if embedded_report_lab_hits >= 2 and strong_table_headers:
        scores["clinical_report_page"] += 6
        scores["lab_table_page"] -= 2

    if mixed_clinical_hits >= 3:
        scores["clinical_report_page"] += 8
        scores["lab_table_page"] -= 2

    if mixed_clinical_hits >= 2 and report_hits >= 2:
        scores["clinical_report_page"] += 6
        scores["lab_table_page"] -= 2

    if (
        mixed_clinical_hits >= 2
        and any(x in joined_norm for x in [
            "evolution et traitement dans le service",
            "traitement de sortie",
            "conclusion",
        ])
    ):
        scores["clinical_report_page"] += 8
        scores["lab_table_page"] -= 3

    if lab_hits >= 6 and report_hits == 0 and embedded_report_lab_hits == 0 and mixed_clinical_hits == 0:
        scores["lab_table_page"] += 6

    if lab_hits >= 8 and report_hits <= 1 and embedded_report_lab_hits == 0 and mixed_clinical_hits == 0:
        scores["lab_table_page"] += 5

    if not layout_flags["has_table_layout"] and not strong_table_headers and not strong_lab_analytes:
        scores["lab_table_page"] = min(scores["lab_table_page"], 1)

    if len(texts) <= 5 and scores["form_page"] > 0 and scores["clinical_report_page"] == 0 and scores["lab_table_page"] == 0:
        scores["form_page"] -= 1

    for k in list(scores.keys()):
        if scores[k] < 0:
            scores[k] = 0

    has_table_markers = lab_hits >= 2
    has_form_markers = (form_hits >= 2) or (cnr_form_hits >= 2)
    has_report_markers = report_hits >= 2
    is_sparse_page = len(texts) <= 5

    positive_scores = {k: v for k, v in scores.items() if v > 0}

    if not positive_scores:
        primary_page_type = "unknown"
        confidence = 0.0
    else:
        best_type, best_score = max(positive_scores.items(), key=lambda kv: kv[1])
        total = sum(positive_scores.values())
        confidence = round(best_score / total, 3) if total else 0.0

        if best_score < 2:
            primary_page_type = "unknown"
        else:
            primary_page_type = best_type

    return {
        "primary_page_type": primary_page_type,
        "confidence": confidence,
        "scores": dict(scores),
        "text_count": len(texts),
        "flags": {
            "has_table_layout": layout_flags["has_table_layout"],
            "has_image_layout": layout_flags["has_image_layout"],
            "has_table_markers": has_table_markers,
            "has_form_markers": has_form_markers,
            "has_report_markers": has_report_markers,
            "is_sparse_page": is_sparse_page,
        },
        "all_text_preview": texts[:20],
        "debug_counts": {
            "lab_hits": lab_hits,
            "form_hits": form_hits,
            "report_hits": report_hits,
            "cnr_form_hits": cnr_form_hits,
            "short_form_hits": short_form_hits,
            "embedded_report_lab_hits": embedded_report_lab_hits,
            "mixed_clinical_hits": mixed_clinical_hits,
            "strong_table_headers": strong_table_headers,
            "strong_lab_analytes": strong_lab_analytes,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Classify document page type from Paddle OCR JSON")
    parser.add_argument("paddle_json", help="Path to Paddle OCR / combined JSON")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification",
        help="Directory to save classification result",
    )
    args = parser.parse_args()

    json_path = Path(args.paddle_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(json_path)
    texts = extract_texts_from_paddle_json(data)
    layout_flags = detect_layout_flags(data)
    result = classify_page(texts, layout_flags)

    out_file = out_dir / f"{json_path.stem}_page_type.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Texts found:        {result['text_count']}")
    print(f"Primary type:       {result['primary_page_type']}")
    print(f"Confidence:         {result['confidence']}")
    print(f"Scores:             {result['scores']}")
    print(f"Flags:              {result['flags']}")
    print(f"Saved result:       {out_file}")


if __name__ == "__main__":
    main()
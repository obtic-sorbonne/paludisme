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


def classify_page(texts, layout_flags):
    joined = "\n".join(texts)
    joined_norm = norm(joined)

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
        "frottis mince",
        "goutte epaisse",
        "parasitemie",
    ]

    form_keywords = [
        "oui",
        "non",
        "nsp",
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
    ]

    report_keywords = [
        "cru",
        "compte rendu",
        "compte rendu des urgences",
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
        "conclusion medicale definitive",
        "modalites de sorties",
        "modalites de sortie",
        "diagnostic principal",
        "diagnostics associes",
        "documents remis au patient",
        "ordonnance(s) de sortie",
        "service des urgences",
        "interne(s)",
        "hospital robert debre",
        "pediatrie generale",
        "motif d'hospitalisation",
    ]

    scores = Counter()

    lab_hits = count_matches(joined_norm, lab_keywords)
    form_hits = count_matches(joined_norm, form_keywords)
    report_hits = count_matches(joined_norm, report_keywords)

    scores["lab_table_page"] += lab_hits
    scores["form_page"] += form_hits
    scores["clinical_report_page"] += report_hits

    if all(k in joined_norm for k in ["description", "resultat", "unite"]):
        scores["lab_table_page"] += 4

    if sum(1 for k in ["oui", "non", "nsp"] if k in joined_norm) >= 2:
        scores["form_page"] += 3

    if "compte rendu" in joined_norm or "cru" in joined_norm:
        scores["clinical_report_page"] += 4

    if count_matches(joined_norm, [
        "prescriptions",
        "evolution",
        "conclusion medicale definitive",
        "diagnostic principal",
        "modalites de sorties",
        "histoire de la maladie",
        "motif d'hospitalisation",
        "antecedents",
    ]) >= 2:
        scores["clinical_report_page"] += 4

    if layout_flags["has_table_layout"] and report_hits >= 2:
        scores["clinical_report_page"] += 1

    if layout_flags["has_table_layout"] and lab_hits >= 2:
        scores["lab_table_page"] += 1

    if len(texts) <= 5 and scores["form_page"] > 0 and scores["clinical_report_page"] == 0 and scores["lab_table_page"] == 0:
        scores["form_page"] -= 1

    has_table_markers = lab_hits >= 2
    has_form_markers = form_hits >= 2
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
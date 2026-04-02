from pathlib import Path
import argparse
import json
import re


def clean_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(data):
    return data.get("res", data) if isinstance(data, dict) else data


def extract_texts_from_paddle_json(data):
    texts = []
    root = get_root(data)

    if isinstance(root, dict):
        ocr = root.get("overall_ocr_res", {})
        if isinstance(ocr, dict):
            rec_texts = ocr.get("rec_texts", [])
            if isinstance(rec_texts, list):
                for t in rec_texts:
                    if isinstance(t, str):
                        t = clean_text(t)
                        if t:
                            texts.append(t)

    return texts


def decide_route(classification: dict):
    primary = classification.get("primary_page_type", "unknown")
    flags = classification.get("flags", {})

    has_table_layout = flags.get("has_table_layout", False)
    is_sparse_page = flags.get("is_sparse_page", False)

    route = {
        "primary_page_type": primary,
        "next_step": None,
        "keep_ocr_text": True,
        "extract_tables": False,
        "extract_form_fields": False,
        "extract_report_text": False,
        "needs_review": False,
        "notes": [],
    }

    if primary == "clinical_report_page":
        route["next_step"] = "report_text_extraction"
        route["extract_report_text"] = True
        if has_table_layout:
            route["extract_tables"] = True
            route["notes"].append("Narrative page also has table layout; keep table metadata.")
        else:
            route["notes"].append("Narrative clinical report page.")

    elif primary == "lab_table_page":
        route["next_step"] = "lab_table_extraction"
        route["extract_tables"] = True
        route["notes"].append("Structured lab/result table page.")

    elif primary == "form_page":
        route["next_step"] = "form_field_extraction"
        route["extract_form_fields"] = True
        if has_table_layout:
            route["extract_tables"] = True
            route["notes"].append("Form page also contains table-like regions.")
        else:
            route["notes"].append("Questionnaire / checkbox-style form page.")

    else:
        route["next_step"] = "keep_text_only"
        route["keep_ocr_text"] = True
        if is_sparse_page:
            route["notes"].append("Sparse or weakly classified page; preserve OCR text.")
        else:
            route["notes"].append("Unknown page type; preserve OCR text and review later.")
        route["needs_review"] = True

    return route


def main():
    parser = argparse.ArgumentParser(description="Route a page to the next processing branch")
    parser.add_argument("paddle_json", help="Path to Paddle page JSON")
    parser.add_argument("classification_json", help="Path to page classification JSON")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page_routing",
        help="Directory to save routing decisions",
    )
    args = parser.parse_args()

    paddle_json_path = Path(args.paddle_json)
    classification_json_path = Path(args.classification_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paddle_data = load_json(paddle_json_path)
    classification = load_json(classification_json_path)

    texts = extract_texts_from_paddle_json(paddle_data)
    route = decide_route(classification)

    result = {
        "source_json": str(paddle_json_path),
        "classification_json": str(classification_json_path),
        "routing": route,
        "text_count": len(texts),
        "text_preview": texts[:15],
    }

    out_file = out_dir / f"{paddle_json_path.stem}_route.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Primary type:       {route['primary_page_type']}")
    print(f"Next step:          {route['next_step']}")
    print(f"Keep OCR text:      {route['keep_ocr_text']}")
    print(f"Extract tables:     {route['extract_tables']}")
    print(f"Extract form:       {route['extract_form_fields']}")
    print(f"Extract report:     {route['extract_report_text']}")
    print(f"Needs review:       {route['needs_review']}")
    print(f"Saved route file:   {out_file}")


if __name__ == "__main__":
    main()
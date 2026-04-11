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
    if not path.exists():
        return {}
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


def main():
    parser = argparse.ArgumentParser(description="Save OCR fallback when lab extraction fails")
    parser.add_argument("paddle_json", help="Path to paddle page json")
    parser.add_argument("classification_json", help="Path to page classification json")
    parser.add_argument("pdf_path", help="Original PDF path")
    parser.add_argument("page_index", type=int, help="0-based page index")
    parser.add_argument("failure_reason", help="Reason for fallback")
    parser.add_argument(
        "--text-output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/lab_fallback_text",
        help="Directory for OCR fallback text files",
    )
    parser.add_argument(
        "--meta-output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/lab_fallback_meta",
        help="Directory for fallback metadata json files",
    )
    args = parser.parse_args()

    paddle_json_path = Path(args.paddle_json)
    classification_json_path = Path(args.classification_json)

    text_out_dir = Path(args.text_output_dir)
    meta_out_dir = Path(args.meta_output_dir)
    text_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    paddle_data = load_json(paddle_json_path)
    classification_data = load_json(classification_json_path)

    texts = extract_texts_from_paddle_json(paddle_data)

    doc_stem = paddle_json_path.stem.replace("_res", "")
    text_out = text_out_dir / f"{doc_stem}_ocr_fallback.txt"
    meta_out = meta_out_dir / f"{doc_stem}_fallback.json"

    with open(text_out, "w", encoding="utf-8") as f:
        f.write(f"PDF: {args.pdf_path}\n")
        f.write(f"Page index: {args.page_index}\n")
        f.write(f"Page num: {args.page_index + 1}\n")
        f.write(f"Failure reason: {args.failure_reason}\n\n")
        f.write("=== OCR TEXT ===\n\n")
        for line in texts:
            f.write(line + "\n")

    result = {
        "pdf_path": args.pdf_path,
        "page_index": args.page_index,
        "page_num": args.page_index + 1,
        "fallback_mode": "keep_text_only",
        "extraction_branch": "lab_table_extraction",
        "extraction_success": False,
        "failure_reason": args.failure_reason,
        "page_type_guess": classification_data.get("primary_page_type", "unknown"),
        "classification_confidence": classification_data.get("confidence", 0.0),
        "classification_scores": classification_data.get("scores", {}),
        "needs_review": True,
        "ocr_fallback_text_file": str(text_out),
        "paddle_json": str(paddle_json_path),
        "classification_json": str(classification_json_path) if classification_json_path.exists() else None,
        "ocr_text_count": len(texts),
        "ocr_preview": texts[:20],
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR fallback text: {text_out}")
    print(f"Saved fallback metadata: {meta_out}")


if __name__ == "__main__":
    main()
from pathlib import Path
import argparse
import re
from paddleocr import PaddleOCR


def clean_line(line: str) -> str:
    line = str(line).replace("\xa0", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def fix_common_ocr_errors(line: str) -> str:
    line = clean_line(line)

    replacements = [
        ("Personnè à prévenir", "Personne à prévenir"),
        ("PA:I", "PA : /"),
        ("PA:l", "PA : /"),
        ("PA: I", "PA : /"),
        ("PA BrasG : I", "PA BrasG : /"),
        ("PA BrasG:I", "PA BrasG : /"),
        ("PA BrasG: I", "PA BrasG : /"),
        ("Actions lA0", "Actions IAO"),
        ("Actions lAO", "Actions IAO"),
        ("Actions IA0", "Actions IAO"),
        ("na parle pas", "ne parle pas"),
        ("Côte d'lvoire", "Côte d'Ivoire"),
        ("Cote d'lvoire", "Cote d'Ivoire"),
        ("momis", "mois"),
        ("oû", "où"),
        ("LARiAM", "LARIAM"),
        ("Télécopie", "Télécopie"),
        ("Personnè", "Personne"),
        ("Caret de santé présenté", "Carnet de santé présenté"),
    ]

    for old, new in replacements:
        line = line.replace(old, new)

    # Normalize common variants around IAO
    line = re.sub(r"\bIA0\b", "IAO", line)
    line = re.sub(r"\blA0\b", "IAO", line)
    line = re.sub(r"\blAO\b", "IAO", line)

    # Normalize Ivory Coast typo if split oddly
    line = re.sub(r"\bd['’]lvoire\b", "d'Ivoire", line)

    # Normalize weird spacing before punctuation
    line = re.sub(r"\s+:", " :", line)
    line = re.sub(r"\s+;", ";", line)

    
    return line


def postprocess_page_lines(lines):
    cleaned = []
    prev = None

    for line in lines:
        line = fix_common_ocr_errors(line)
        if not line:
            continue

        # remove exact consecutive duplicates
        if line == prev:
            continue

        cleaned.append(line)
        prev = line

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR on one PDF")
    parser.add_argument("pdf_path", help="Full path to the PDF file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle",
        help="Directory where output text file will be saved",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="fr",
    )

    result = ocr.predict(str(pdf_path))
    out_file = out_dir / (pdf_path.stem + ".txt")

    with open(out_file, "w", encoding="utf-8") as f:
        for page_idx, page in enumerate(result, start=1):
            f.write(f"===== PAGE {page_idx} =====\n\n")

            page_lines = []

            if isinstance(page, dict):
                rec_texts = page.get("rec_texts", [])
                if rec_texts:
                    for line in rec_texts:
                        page_lines.append(str(line))
                else:
                    page_lines.append(str(page))
            else:
                page_lines.append(str(page))

            page_lines = postprocess_page_lines(page_lines)

            for line in page_lines:
                f.write(line + "\n")

            f.write("\n")

    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
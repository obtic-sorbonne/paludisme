import csv
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from . import name_extraction, anonymizer

logger = logging.getLogger(__name__)
nlp = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_texts_from_folder(input_dir: Path) -> dict[str, str]:
    texts = {}
    for path in sorted(input_dir.glob("*.txt")):
        texts[path.name] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def process_folder(input_dir: Path, output_dir: Path, patient_id: str = "001"):
    doc_texts = load_texts_from_folder(input_dir)
    if not doc_texts:
        logger.error("No .txt files found in %s", input_dir)
        return

    patient_info = name_extraction.extract(list(doc_texts.values()))
    if not patient_info:
        logger.error("Could not extract patient name from %s", input_dir)
        return

    out_dir = output_dir / input_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_replacements = []
    total_stats = defaultdict(int)
    merged_parts = []

    for filename, text in doc_texts.items():
        anon_text, stats, replacements = anonymizer.anonymize(
            text=text,
            patient_info=patient_info,
            patient_id=patient_id,
            nlp=nlp,
        )

        merged_parts.append(f"\n{'='*70}\n{filename}\n{'='*70}\n\n{anon_text}")

        out_txt = out_dir / filename
        out_txt.write_text(anon_text, encoding="utf-8")

        for k, v in stats.items():
            total_stats[k] += v

        for r in replacements:
            all_replacements.append({
                "file": filename,
                "category": r["category"],
                "original": r["original"],
                "replacement": r["replacement"],
                "start": r["start"],
                "end": r["end"],
            })

    (out_dir / "all_documents_anonymized.txt").write_text(
        "\n".join(merged_parts), encoding="utf-8"
    )

    with open(out_dir / "replacements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "category", "original", "replacement", "start", "end"],
        )
        writer.writeheader()
        writer.writerows(all_replacements)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Patient ID: PATIENT_{patient_id}\n")
        f.write(f"Folder: {input_dir.name}\n\n")
        f.write("Stats:\n")
        for k, v in sorted(total_stats.items()):
            f.write(f"  {k}: {v}\n")

    logger.info("Done: %s", out_dir)


def run(input_root: Path, output_root: Path):
    setup_logging()

    try:
        import spacy
        global nlp
        nlp = spacy.load("fr_core_news_lg")
        logger.info("spaCy loaded.")
    except Exception:
        nlp = None
        logger.warning("spaCy not available. Running regex-only mode.")

    folders = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not folders:
        logger.error("No subfolders found in %s", input_root)
        return

    for idx, folder in enumerate(folders, start=1):
        process_folder(folder, output_root, patient_id=f"{idx:03d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="root folder containing subfolders of OCR txt files")
    parser.add_argument("-o", "--output", required=True, help="output root")
    args = parser.parse_args()

    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
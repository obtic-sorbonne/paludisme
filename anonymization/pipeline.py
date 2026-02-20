"""
Main pipeline — orchestrates OCR, name extraction, and anonymization
across patient subfolders.

Usage:
    python -m anonymization -i /path/to/root -o /path/to/output
    python -m anonymization -i /path/to/root -o /path/to/output --gpu
"""

import csv
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

from . import ocr, name_extraction, pseudonyms, anonymizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to load spaCy (optional but recommended)
try:
    import spacy
    nlp = spacy.load("fr_core_news_lg")
    logger.info("spaCy fr_core_news_lg loaded — NER enabled.")
except Exception:
    nlp = None
    logger.warning(
        "spaCy not available — regex-only mode. "
        "Install: pip install spacy && python -m spacy download fr_core_news_lg"
    )


def process_subfolder(
    subfolder: Path,
    output_dir: Path,
    patient_id: str = "001",
    use_gpu: bool = False,
    artifacts_path: str = None,
) -> Optional[dict]:
    """Process all PDFs in one patient subfolder."""
    pdf_files = sorted(
        f for f in subfolder.iterdir()
        if f.suffix.lower() == ".pdf" and f.is_file()
    )
    if not pdf_files:
        logger.warning(f"No PDFs in {subfolder.name}")
        return None

    logger.info(f"Processing: {subfolder.name} ({len(pdf_files)} PDFs)")

    # 1. OCR
    doc_texts = ocr.process_folder(
        subfolder, use_gpu=use_gpu, artifacts_path=artifacts_path
    )

    # 2. Extract patient name
    patient_info = name_extraction.extract(list(doc_texts.values()))
    if not patient_info:
        logger.error(f"  Cannot identify patient in {subfolder.name} — skipping.")
        return None
    logger.info(f"  Patient: {patient_info['lastnames']} {patient_info['firstnames']}")

    # 3. Generate pseudonyms
    pseudo_map = pseudonyms.generate(patient_info, patient_id=patient_id)

    # 4. Anonymize and concatenate
    total_stats = defaultdict(int)
    all_replacements = []  # (doc_name, original, replacement, category)
    parts = []
    for doc_name, text in doc_texts.items():
        header = f"\n{'='*60}\n{doc_name}\n{'='*60}\n\n"
        anon_text, stats, replacements = anonymizer.anonymize(text, patient_info, pseudo_map, nlp)
        parts.append(header + anon_text)
        for k, v in stats.items():
            total_stats[k] += v
        for r in replacements:
            all_replacements.append({
                "file": doc_name,
                "original": r["original"],
                "replacement": r["replacement"],
                "category": r["category"],
            })

    # 5. Write output
    out_sub = output_dir / subfolder.name
    out_sub.mkdir(parents=True, exist_ok=True)
    (out_sub / "all_documents.txt").write_text("\n".join(parts), encoding="utf-8")

    # Detailed replacements log
    repl_path = out_sub / "replacements.csv"
    with open(repl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "category", "original", "replacement"])
        writer.writeheader()
        writer.writerows(all_replacements)

    logger.info(f"  Stats: {dict(total_stats)}")
    logger.info(f"  Replacements log: {repl_path}")

    return {
        "subfolder": subfolder.name,
        "patient_id": f"PATIENT_{patient_id}",
        "real_lastnames": " ".join(patient_info["lastnames"]),
        "real_firstnames": " ".join(patient_info["firstnames"]),
        "num_docs": len(pdf_files),
        "stats": dict(total_stats),
    }


def run(
    input_dir: Path,
    output_dir: Path,
    use_gpu: bool = False,
    artifacts_path: str = None,
):
    """Run full pipeline on all patient subfolders."""
    output_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and any(f.suffix.lower() == ".pdf" for f in d.iterdir())
    )
    if not subfolders:
        logger.error(f"No subfolders with PDFs in {input_dir}")
        return

    logger.info(f"Found {len(subfolders)} patient subfolder(s).")

    mappings = []
    for idx, sf in enumerate(subfolders, start=1):
        patient_id = f"{idx:03d}"
        result = process_subfolder(
            sf, output_dir, patient_id=patient_id,
            use_gpu=use_gpu, artifacts_path=artifacts_path
        )
        if result:
            mappings.append(result)

    # Mapping CSV
    csv_path = output_dir / "mapping.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subfolder", "patient_id", "real_lastnames", "real_firstnames",
            "num_docs",
        ])
        writer.writeheader()
        for m in mappings:
            writer.writerow({k: v for k, v in m.items() if k != "stats"})

    # Report
    report_path = output_dir / "anonymization_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ANONYMIZATION REPORT\n" + "=" * 60 + "\n\n")
        for m in mappings:
            f.write(f"Folder: {m['subfolder']}\n")
            f.write(f"  Real:      {m['real_lastnames']} {m['real_firstnames']}\n")
            f.write(f"  Replaced:  [{m['patient_id']}]\n")
            f.write(f"  Docs: {m['num_docs']}\n")
            for k, v in m["stats"].items():
                f.write(f"    {k}: {v}\n")
            f.write("\n")

    logger.info(f"CSV: {csv_path}")
    logger.info(f"Report: {report_path}")
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Anonymize French medical PDFs.")
    parser.add_argument("--input", "-i", required=True, help="Root folder with patient subfolders")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (CUDA)")
    parser.add_argument(
        "--artifacts-path",
        default=None,
        help="Path to pre-downloaded Docling models (for offline use)",
    )
    args = parser.parse_args()
    run(
        Path(args.input),
        Path(args.output),
        use_gpu=args.gpu,
        artifacts_path=args.artifacts_path,
    )


if __name__ == "__main__":
    main()
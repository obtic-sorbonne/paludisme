import argparse
import logging
from pathlib import Path

from . import name_extraction, anonymizer

logger = logging.getLogger(__name__)
nlp = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Test anonymization on one OCR txt file")
    parser.add_argument("input_txt", help="Path to OCR txt file")
    parser.add_argument("-o", "--output", required=True, help="Output anonymized txt path")
    parser.add_argument("--patient-id", default="001", help="Patient ID number")
    args = parser.parse_args()

    setup_logging()

    global nlp
    try:
        import spacy
        nlp = spacy.load("fr_core_news_lg")
        logger.info("spaCy loaded.")
    except Exception:
        nlp = None
        logger.warning("spaCy not available. Running regex-only mode.")

    input_path = Path(args.input_txt)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8", errors="ignore")

    patient_info = name_extraction.extract([text])
    if not patient_info:
        logger.error("Could not extract patient name from file: %s", input_path)
        return

    anon_text, stats, replacements = anonymizer.anonymize(
        text=text,
        patient_info=patient_info,
        patient_id=args.patient_id,
        nlp=nlp,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(anon_text, encoding="utf-8")

    repl_path = output_path.with_suffix(".replacements.txt")
    with repl_path.open("w", encoding="utf-8") as f:
        f.write("REPLACEMENTS\n")
        f.write("=" * 60 + "\n\n")
        for r in replacements:
            f.write(
                f"{r['category']}: {r['original']} -> {r['replacement']} "
                f"(span={r['start']}:{r['end']})\n"
            )

    logger.info("Anonymized file written to: %s", output_path)
    logger.info("Replacement log written to: %s", repl_path)
    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
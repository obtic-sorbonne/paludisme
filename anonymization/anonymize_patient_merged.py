from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from . import anonymizer, name_extraction

logger = logging.getLogger(__name__)
nlp = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def anonymize_string(
    text: str,
    patient_info: dict,
    patient_id: str,
) -> tuple[str, dict, list[dict]]:
    return anonymizer.anonymize(
        text=text,
        patient_info=patient_info,
        patient_id=patient_id,
        nlp=nlp,
    )


def anonymize_json_obj(obj: Any, patient_info: dict, patient_id: str) -> Any:
    """
    Recursively anonymize every string value inside a JSON object.
    """
    if isinstance(obj, dict):
        return {
            key: anonymize_json_obj(value, patient_info, patient_id)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [anonymize_json_obj(item, patient_info, patient_id) for item in obj]

    if isinstance(obj, str):
        anon_text, _, _ = anonymize_string(obj, patient_info, patient_id)
        return anon_text

    return obj


def write_replacements_csv(path: Path, replacements: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "original", "replacement", "start", "end"],
        )
        writer.writeheader()
        writer.writerows(replacements)


def write_summary(
    path: Path,
    input_txt: Path,
    patient_info: dict,
    patient_id: str,
    stats: dict,
    replacements: list[dict],
) -> None:
    lines = []
    lines.append(f"Input file: {input_txt}")
    lines.append(f"Patient ID tag: PATIENT_{patient_id}")
    lines.append("")

    if patient_info:
        lines.append("Extracted patient info:")
        lines.append(f"  Lastnames:  {' '.join(patient_info.get('lastnames', []))}")
        lines.append(f"  Firstnames: {' '.join(patient_info.get('firstnames', []))}")
        lines.append(f"  Tokens:     {patient_info.get('all_tokens', [])}")
        lines.append("")

    lines.append("Stats:")
    if stats:
        for k, v in sorted(stats.items()):
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  No replacements made.")

    lines.append("")
    lines.append(f"Total replacements: {len(replacements)}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anonymize one patient merged TXT file and optionally its matching JSON."
    )
    parser.add_argument(
        "merged_txt",
        help="Path to one patient merged TXT file, e.g. benchmark_outputs/patient_merged_records/2006_RDB_0156_merged.txt",
    )
    parser.add_argument(
        "--patient-id",
        default="001",
        help="Patient ID number used in replacement tag, default: 001",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: same folder as input under anonymized/",
    )
    parser.add_argument(
        "--with-json",
        action="store_true",
        help="Also anonymize the matching _merged.json file if present.",
    )
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

    merged_txt_path = Path(args.merged_txt).expanduser().resolve()
    if not merged_txt_path.exists():
        raise FileNotFoundError(f"Merged TXT not found: {merged_txt_path}")

    merged_text = load_text(merged_txt_path)

    patient_info = name_extraction.extract([merged_text])
    if not patient_info:
        raise RuntimeError(
            f"Could not extract patient name from merged file: {merged_txt_path}"
        )

    anon_text, stats, replacements = anonymize_string(
        merged_text,
        patient_info,
        args.patient_id,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = merged_txt_path.parent / "anonymized"

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = merged_txt_path.stem.replace("_merged", "")
    anon_txt_path = out_dir / f"{stem}_merged_anonymized.txt"
    repl_csv_path = out_dir / f"{stem}_replacements.csv"
    summary_path = out_dir / f"{stem}_summary.txt"

    anon_txt_path.write_text(anon_text, encoding="utf-8")
    write_replacements_csv(repl_csv_path, replacements)
    write_summary(
        summary_path,
        merged_txt_path,
        patient_info,
        args.patient_id,
        stats,
        replacements,
    )

    logger.info("Anonymized TXT written to: %s", anon_txt_path)
    logger.info("Replacement CSV written to: %s", repl_csv_path)
    logger.info("Summary written to: %s", summary_path)

    if args.with_json:
        merged_json_path = merged_txt_path.with_suffix(".json")
        if merged_json_path.exists():
            merged_json = load_json(merged_json_path)
            anon_json = anonymize_json_obj(merged_json, patient_info, args.patient_id)
            anon_json_path = out_dir / f"{stem}_merged_anonymized.json"
            save_json(anon_json_path, anon_json)
            logger.info("Anonymized JSON written to: %s", anon_json_path)
        else:
            logger.warning("Matching JSON not found, skipped: %s", merged_json_path)


if __name__ == "__main__":
    main()
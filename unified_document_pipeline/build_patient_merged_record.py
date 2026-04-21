from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path("/home/lfarooq/digitize_medical_records")
BENCHMARK_DIR = ROOT_DIR / "benchmark_outputs"
FINAL_DOC_DIR = BENCHMARK_DIR / "final_document_pipeline"
PATIENT_MERGED_DIR = BENCHMARK_DIR / "patient_merged_records"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def normalize_date(dd: str, mm: str, yyyy: str) -> str:
    return f"{yyyy}-{mm}-{dd}"


def extract_all_dates(text: str) -> List[str]:
    matches = re.findall(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", text or "")
    return sorted({normalize_date(dd, mm, yyyy) for dd, mm, yyyy in matches})


def detect_document_type(merged_json: Dict[str, Any]) -> str:
    next_steps = [page.get("next_step", "") for page in merged_json.get("pages", [])]

    has_form = "form_field_extraction" in next_steps
    has_lab = "lab_table_extraction" in next_steps
    has_report = "report_text_extraction" in next_steps or "keep_text_only" in next_steps

    if has_form and not has_report and not has_lab:
        return "checkbox_form"
    if has_lab and not has_form and not has_report:
        return "lab_table"
    if has_lab and has_form:
        return "mixed_form_lab"
    if has_form:
        return "consultation_form"
    if has_report:
        return "clinical_report"
    return "unknown"


def extract_relevant_event_dates(merged_text: str) -> List[str]:
    """
    Prefer only dates associated with real clinical events.
    """
    text = merged_text or ""
    found = set()

    patterns = [
        r"Date\s*:\s*(\d{2})/(\d{2})/(\d{4})",
        r"Date de la consultation actuelle\s*:\s*(\d{2})/(\d{2})/(\d{4})",
        r"Date du Diagnostic Biologique\s*[:：]?\s*(\d{2})/(\d{2})/(\d{4})",
        r"PARAMETRES A L['’]ARRIVEE\s*:\s*(\d{2})/(\d{2})/(\d{4})",
        r"Pr[ée]l[eè]vement effectu[eé] le\s*(\d{2})/(\d{2})/(\d{4})",
        r"consultation le\s*(\d{2})/(\d{2})/(\d{4})",
        r"Contr[ôo]le le\s*(\d{2})/(\d{2})/(\d{4})",
        r"Consulte [àa].*? le\s*(\d{2})/(\d{2})/(\d{4})",
        r"se pr[eé]sente .*? le\s*(\d{2})/(\d{2})/(\d{4})",
    ]

    for pat in patterns:
        for dd, mm, yyyy in re.findall(pat, text, flags=re.IGNORECASE):
            found.add(normalize_date(dd, mm, yyyy))

    return sorted(found)


def detect_document_dates(merged_json: Dict[str, Any], merged_txt: str) -> Dict[str, List[str]]:
    relevant_dates = extract_relevant_event_dates(merged_txt)
    all_dates = extract_all_dates(merged_txt)

    return {
        "relevant_dates": relevant_dates,
        "all_dates": all_dates,
    }


def assign_timepoints_from_relevant_dates(unique_dates: List[str]) -> Dict[str, Optional[str]]:
    unique_dates = sorted(unique_dates)
    mapping = {"J0": None, "J3": None, "J30": None}

    if len(unique_dates) > 0:
        mapping["J0"] = unique_dates[0]
    if len(unique_dates) > 1:
        mapping["J3"] = unique_dates[1]
    if len(unique_dates) > 2:
        mapping["J30"] = unique_dates[2]

    return mapping


def build_patient_record(patient_folder: Path) -> Dict[str, Any]:
    pdf_paths = sorted(patient_folder.rglob("*.pdf"))

    documents: List[Dict[str, Any]] = []
    missing_outputs: List[str] = []
    all_relevant_dates = set()
    all_dates = set()

    for pdf_path in pdf_paths:
        doc_stem = pdf_path.stem
        doc_dir = FINAL_DOC_DIR / doc_stem
        merged_json_path = doc_dir / "merged_final_output.json"
        merged_txt_path = doc_dir / "merged_final_output.txt"

        if not merged_json_path.exists() or not merged_txt_path.exists():
            missing_outputs.append(str(pdf_path))
            continue

        merged_json = load_json(merged_json_path)
        merged_txt = load_text(merged_txt_path)

        doc_type = detect_document_type(merged_json)
        date_info = detect_document_dates(merged_json, merged_txt)

        relevant_dates = date_info["relevant_dates"]
        all_doc_dates = date_info["all_dates"]

        for d in relevant_dates:
            all_relevant_dates.add(d)
        for d in all_doc_dates:
            all_dates.add(d)

        documents.append(
            {
                "doc_stem": doc_stem,
                "pdf_path": str(pdf_path),
                "doc_type": doc_type,
                "merged_json_path": str(merged_json_path),
                "merged_txt_path": str(merged_txt_path),
                "relevant_dates": relevant_dates,
                "all_detected_dates": all_doc_dates,
                "merged_text": merged_txt,
                "merged_json": merged_json,
            }
        )

    unique_relevant_dates = sorted(all_relevant_dates)
    timepoint_dates = assign_timepoints_from_relevant_dates(unique_relevant_dates)

    timepoints = {
        "J0": {"date": timepoint_dates["J0"], "documents": []},
        "J3": {"date": timepoint_dates["J3"], "documents": []},
        "J30": {"date": timepoint_dates["J30"], "documents": []},
    }

    for doc in documents:
        doc_dates = set(doc["relevant_dates"])
        for tp in ["J0", "J3", "J30"]:
            tp_date = timepoints[tp]["date"]
            if tp_date and tp_date in doc_dates:
                timepoints[tp]["documents"].append(doc["doc_stem"])

    return {
        "patient_folder": str(patient_folder),
        "pdf_count": len(pdf_paths),
        "documents_with_outputs": len(documents),
        "missing_outputs": missing_outputs,
        "unique_relevant_dates_found": unique_relevant_dates,
        "all_unique_dates_found": sorted(all_dates),
        "timepoints": timepoints,
        "documents": documents,
    }


def write_patient_outputs(patient_folder: Path, record: Dict[str, Any]) -> None:
    PATIENT_MERGED_DIR.mkdir(parents=True, exist_ok=True)

    patient_name = patient_folder.name.replace(" ", "_")
    json_path = PATIENT_MERGED_DIR / f"{patient_name}_merged.json"
    txt_path = PATIENT_MERGED_DIR / f"{patient_name}_merged.txt"

    save_json(json_path, record)

    lines: List[str] = []
    lines.append(f"PATIENT_FOLDER: {record['patient_folder']}")
    lines.append(f"PDF_COUNT: {record['pdf_count']}")
    lines.append(f"DOCUMENTS_WITH_OUTPUTS: {record['documents_with_outputs']}")
    lines.append("")

    lines.append("UNIQUE_RELEVANT_DATES_FOUND:")
    for d in record["unique_relevant_dates_found"]:
        lines.append(f"  - {d}")

    lines.append("")
    lines.append("ALL_UNIQUE_DATES_FOUND:")
    for d in record["all_unique_dates_found"]:
        lines.append(f"  - {d}")

    lines.append("")
    lines.append("TIMEPOINTS:")
    for tp in ["J0", "J3", "J30"]:
        tp_info = record["timepoints"][tp]
        lines.append(f"  {tp}: {tp_info['date']}")
        for doc_stem in tp_info["documents"]:
            lines.append(f"    - {doc_stem}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("FULL MERGED DOCUMENT CONTENT")
    lines.append("=" * 100)
    lines.append("")

    for doc in record["documents"]:
        lines.append("=" * 100)
        lines.append(f"DOCUMENT: {doc['doc_stem']}")
        lines.append(f"TYPE: {doc['doc_type']}")
        lines.append(f"PDF: {doc['pdf_path']}")
        lines.append(f"RELEVANT_DATES: {doc['relevant_dates']}")
        lines.append(f"ALL_DETECTED_DATES: {doc['all_detected_dates']}")
        lines.append(f"SOURCE_TXT: {doc['merged_txt_path']}")
        lines.append("=" * 100)
        lines.append(doc["merged_text"].strip())
        lines.append("")
        lines.append("")

    if record["missing_outputs"]:
        lines.append("=" * 100)
        lines.append("MISSING_OUTPUTS")
        lines.append("=" * 100)
        for p in record["missing_outputs"]:
            lines.append(f"  - {p}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved patient merged JSON: {json_path}")
    print(f"Saved patient merged TXT:  {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build one true merged patient record from final document pipeline outputs."
    )
    parser.add_argument("patient_folder", help="Path to one patient folder")
    args = parser.parse_args()

    patient_folder = Path(args.patient_folder).expanduser().resolve()
    if not patient_folder.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_folder}")

    record = build_patient_record(patient_folder)
    write_patient_outputs(patient_folder, record)


if __name__ == "__main__":
    main()
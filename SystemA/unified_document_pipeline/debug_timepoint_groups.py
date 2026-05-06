from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_final_scientist_table import (
    load_json,
    assign_timepoints_from_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug J0/J3/J30 grouping for a patient merged JSON.")
    parser.add_argument("patient_merged_json", help="Path to patient merged JSON")
    args = parser.parse_args()

    patient_json_path = Path(args.patient_merged_json).expanduser().resolve()
    if not patient_json_path.exists():
        raise FileNotFoundError(f"Patient merged JSON not found: {patient_json_path}")

    record = load_json(patient_json_path)
    tp_map = assign_timepoints_from_groups(record)

    out = {}
    for key in ["J0", "J3", "J30"]:
        group = tp_map.get(key, {}).get("group")
        out[key] = {
            "date": tp_map.get(key, {}).get("date", ""),
            "documents": tp_map.get(key, {}).get("documents", []),
            "clinical_docs": [d.get("doc_stem", "") for d in (group or {}).get("clinical_docs", [])],
            "lab_docs": [d.get("doc_stem", "") for d in (group or {}).get("lab_docs", [])],
            "form_docs": [d.get("doc_stem", "") for d in (group or {}).get("form_docs", [])],
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
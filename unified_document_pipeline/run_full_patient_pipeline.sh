#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/patient_folder"
  exit 1
fi

PATIENT_FOLDER="$1"

if [ ! -d "$PATIENT_FOLDER" ]; then
  echo "ERROR: Folder not found: $PATIENT_FOLDER"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "=============================="
echo "FULL PATIENT PIPELINE"
echo "Patient folder: $PATIENT_FOLDER"
echo "=============================="

find "$PATIENT_FOLDER" -type f -iname "*.pdf" | sort | while IFS= read -r pdf; do
  echo "------------------------------"
  echo "Running document pipeline on:"
  echo "$pdf"
  echo "------------------------------"

  python3 unified_document_pipeline/document_pipeline.py "$pdf"
done

echo ""
echo "=============================="
echo "Building patient merged record"
echo "=============================="
python3 unified_document_pipeline/build_patient_merged_record.py "$PATIENT_FOLDER"

PATIENT_NAME="$(basename "$PATIENT_FOLDER" | tr ' ' '_')"
MERGED_JSON="benchmark_outputs/patient_merged_records/${PATIENT_NAME}_merged.json"
MERGED_TXT="benchmark_outputs/patient_merged_records/${PATIENT_NAME}_merged.txt"

# Extract numeric patient id from folder name, e.g. 2006_RDB_0186 -> 186
PATIENT_ID="$(echo "$PATIENT_NAME" | sed -E 's/.*_0*([0-9]+)$/\1/')"

echo ""
echo "=============================="
echo "Extracting final scientist table row"
echo "=============================="
python3 unified_document_pipeline/extract_final_scientist_table.py "$MERGED_JSON"

echo ""
echo "=============================="
echo "Running anonymization"
echo "=============================="
python3 -m anonymization.anonymize_patient_merged \
"$MERGED_TXT" \
--patient-id "$PATIENT_ID" \
--with-json

echo ""
echo "Done."
echo "Merged JSON:        $MERGED_JSON"
echo "Merged TXT:         $MERGED_TXT"
echo "Excel:              benchmark_outputs/final_scientist_table/final_scientist_table.xlsx"
echo "Anonymized output:  benchmark_outputs/patient_merged_records/anonymized/${PATIENT_NAME}_merged_anonymized.txt"
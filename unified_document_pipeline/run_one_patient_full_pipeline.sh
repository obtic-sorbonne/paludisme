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

echo "=================================================="
echo "FULL PATIENT PIPELINE"
echo "Patient folder: $PATIENT_FOLDER"
echo "=================================================="

# Step 1: run document pipeline for all PDFs in this patient folder
find "$PATIENT_FOLDER" -type f -iname "*.pdf" | sort | while IFS= read -r pdf; do
  doc_stem="$(basename "$pdf" .pdf)"

  echo "------------------------------"
  echo "Processing PDF: $pdf"
  echo "DOC_STEM: $doc_stem"
  echo "------------------------------"

  rm -rf "benchmark_outputs/final_document_pipeline/$doc_stem"
  rm -f benchmark_outputs/page_routing/${doc_stem}_*_res_route.json
  rm -f benchmark_outputs/page_classification/${doc_stem}_*_page_type.json

  if ! python3 unified_document_pipeline/document_pipeline.py "$pdf"; then
    echo "FAILED PDF: $pdf"
  fi

  echo ""
done

# Step 2: build merged patient record
python3 unified_document_pipeline/build_patient_merged_record.py "$PATIENT_FOLDER"

# Step 3: extract final scientist row and update Excel
PATIENT_NAME="$(basename "$PATIENT_FOLDER" | sed 's/ /_/g')"
MERGED_JSON="benchmark_outputs/patient_merged_records/${PATIENT_NAME}_merged.json"

if [ -f "$MERGED_JSON" ]; then
  python3 unified_document_pipeline/extract_final_scientist_table.py "$MERGED_JSON"
else
  echo "ERROR: merged JSON not found: $MERGED_JSON"
  exit 1
fi

echo ""
echo "Done."
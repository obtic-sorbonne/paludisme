#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/folder_containing_patient_folders"
  exit 1
fi

ROOT_PATIENT_DIR="$1"

if [ ! -d "$ROOT_PATIENT_DIR" ]; then
  echo "ERROR: Folder not found: $ROOT_PATIENT_DIR"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "========================================"
echo "Batch patient pipeline starting"
echo "Root folder: $ROOT_PATIENT_DIR"
echo "========================================"

find "$ROOT_PATIENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r PATIENT_FOLDER; do
  PATIENT_NAME="$(basename "$PATIENT_FOLDER")"
  PATIENT_SAFE_NAME="${PATIENT_NAME// /_}"

  echo ""
  echo "############################################################"
  echo "PATIENT FOLDER: $PATIENT_FOLDER"
  echo "############################################################"

  PDF_COUNT=$(find "$PATIENT_FOLDER" -type f -iname "*.pdf" | wc -l)

  if [ "$PDF_COUNT" -eq 0 ]; then
    echo "No PDFs found in: $PATIENT_FOLDER"
    continue
  fi

  echo "Step 1/3: Running document pipeline for all PDFs..."
  find "$PATIENT_FOLDER" -type f -iname "*.pdf" | sort | while IFS= read -r pdf; do
    doc_stem="$(basename "$pdf" .pdf)"

    echo "------------------------------------------------------------"
    echo "Processing PDF: $pdf"
    echo "DOC_STEM: $doc_stem"
    echo "------------------------------------------------------------"

    rm -rf "benchmark_outputs/final_document_pipeline/$doc_stem"
    rm -f benchmark_outputs/page_routing/${doc_stem}_*_res_route.json
    rm -f benchmark_outputs/page_classification/${doc_stem}_*_page_type.json

    if ! python3 unified_document_pipeline/document_pipeline.py "$pdf"; then
      echo "FAILED PDF: $pdf"
    fi
  done

  echo ""
  echo "Step 2/3: Building merged patient record..."
  python3 unified_document_pipeline/build_patient_merged_record.py "$PATIENT_FOLDER"

  PATIENT_JSON="benchmark_outputs/patient_merged_records/${PATIENT_SAFE_NAME}_merged.json"

  if [ ! -f "$PATIENT_JSON" ]; then
    echo "Merged patient JSON not found: $PATIENT_JSON"
    continue
  fi

  echo ""
  echo "Step 3/3: Extracting final scientist table row and updating Excel..."
  python3 unified_document_pipeline/extract_final_scientist_table.py "$PATIENT_JSON"

  echo ""
  echo "Done for patient: $PATIENT_NAME"
done

echo ""
echo "========================================"
echo "All patient folders processed."
echo "Excel file updated at:"
echo "~/digitize_medical_records/benchmark_outputs/final_scientist_table/final_scientist_table.xlsx"
echo "========================================"
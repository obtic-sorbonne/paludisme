#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/folder_containing_patient_folders"
  exit 1
fi

PARENT_FOLDER="$1"

if [ ! -d "$PARENT_FOLDER" ]; then
  echo "ERROR: Folder not found: $PARENT_FOLDER"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "=================================================="
echo "BATCH ALL PATIENTS PIPELINE"
echo "Parent folder: $PARENT_FOLDER"
echo "=================================================="

find "$PARENT_FOLDER" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r PATIENT_FOLDER; do
  echo ""
  echo "##################################################"
  echo "PATIENT: $PATIENT_FOLDER"
  echo "##################################################"

  bash unified_document_pipeline/run_one_patient_full_pipeline.sh "$PATIENT_FOLDER"
done

echo ""
echo "=================================================="
echo "ALL PATIENTS DONE"
echo "Excel file:"
echo "~/digitize_medical_records/benchmark_outputs/final_scientist_table/final_scientist_table.xlsx"
echo "=================================================="
#!/usr/bin/env bash
set -u

PDF_DIR="${1:-}"

if [ -z "$PDF_DIR" ]; then
  echo "Usage: ./narrative_clinical_pipeline/batch_run_narrative_layout_pipeline.sh /full/path/to/pdf_dir"
  exit 1
fi

if [ ! -d "$PDF_DIR" ]; then
  echo "Error: directory not found: $PDF_DIR"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

success_count=0
fail_count=0

shopt -s nullglob
pdfs=("$PDF_DIR"/*.pdf)

if [ ${#pdfs[@]} -eq 0 ]; then
  echo "No PDF files found in: $PDF_DIR"
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  echo "========================================"
  echo "Processing PDF: $pdf"
  echo "========================================"

  if ./narrative_clinical_pipeline/run_narrative_layout_pipeline.sh "$pdf"; then
    echo "SUCCESS: $pdf"
    success_count=$((success_count + 1))
  else
    echo "FAILED:  $pdf"
    fail_count=$((fail_count + 1))
  fi
done

echo "========================================"
echo "Batch finished."
echo "Succeeded: $success_count"
echo "Failed:    $fail_count"
echo "========================================"

if [ "$fail_count" -gt 0 ]; then
  exit 1
fi

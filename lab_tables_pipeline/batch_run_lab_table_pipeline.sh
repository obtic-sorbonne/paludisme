#!/usr/bin/env bash
set -u

PDF_DIR="${1:-}"
PAGE_INDEX="${2:-0}"

if [ -z "$PDF_DIR" ]; then
  echo "Usage: ./lab_tables_pipeline/batch_run_lab_table_pipeline.sh /full/path/to/pdf_dir [page_index]"
  exit 1
fi

if [ ! -d "$PDF_DIR" ]; then
  echo "Error: directory not found: $PDF_DIR"
  exit 1
fi

if ! [[ "$PAGE_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Error: page_index must be a non-negative integer"
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

  if ./lab_tables_pipeline/run_lab_table_pipeline.sh "$pdf" "$PAGE_INDEX"; then
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
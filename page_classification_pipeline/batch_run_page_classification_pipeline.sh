#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /absolute/path/to/folder"
  exit 1
fi

PDF_DIR="$1"

if [ ! -d "$PDF_DIR" ]; then
  echo "ERROR: Folder not found: $PDF_DIR"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate

echo "=============================="
echo "Batch page classification pipeline"
echo "Processing folder: $PDF_DIR"
echo "=============================="

shopt -s nullglob
PDFS=("$PDF_DIR"/*.pdf)

if [ ${#PDFS[@]} -eq 0 ]; then
  echo "ERROR: No PDF files found in: $PDF_DIR"
  exit 1
fi

for pdf in "${PDFS[@]}"; do
  DOC_STEM=$(basename "$pdf" .pdf)

  echo ""
  echo "------------------------------"
  echo "Processing PDF: $pdf"
  echo "DOC_STEM: $DOC_STEM"
  echo "------------------------------"

  # 1) Create per-page Paddle JSON
  python page_classification_pipeline/table_paddle_test.py "$pdf"

  # 2) Classify each generated page JSON
  for json in /home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/${DOC_STEM}_*_res.json; do
    [ -e "$json" ] || continue

    echo "Classifying: $json"
    python page_classification_pipeline/classify_page_type.py "$json"

    class_json="/home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification/$(basename "$json" .json)_page_type.json"

    echo "Routing: $json"
    python page_classification_pipeline/route_page_processing.py "$json" "$class_json"
  done

  echo "Finished: $DOC_STEM"
done

echo ""
echo "Batch run complete."
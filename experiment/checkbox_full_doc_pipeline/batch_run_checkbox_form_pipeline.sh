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
echo "Batch checkbox/form pipeline"
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
  python table_paddle_test.py "$pdf"

  # 2) Render page PNGs for visual fallback
  mkdir -p /home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages
  pdftoppm -png "$pdf" "/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages/${DOC_STEM}"

  # 3) Create OCR TXT for the full document
  python narrative_clinical_pipeline/ocr_paddle_test.py "$pdf"

  # 4) Run full document form parser
  python checkbox_full_doc_pipeline/parse_full_document_all_pages.py \
  "/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle/${DOC_STEM}.txt"

  echo "Finished: $DOC_STEM"
done

echo ""
echo "Batch run complete."
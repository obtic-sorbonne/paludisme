#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /absolute/path/to/file.pdf"
  exit 1
fi

PDF="$1"

if [ ! -f "$PDF" ]; then
  echo "ERROR: PDF not found: $PDF"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate

DOC_STEM=$(basename "$PDF" .pdf)

echo "=============================="
echo "Checkbox/Form pipeline"
echo "Processing PDF: $PDF"
echo "DOC_STEM: $DOC_STEM"
echo "=============================="

# 1) Create per-page Paddle JSON
python page_classification_pipeline/table_paddle_test.py "$PDF"

# 2) Render page PNGs for visual fallback
mkdir -p /home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages
pdftoppm -png "$PDF" "/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages/${DOC_STEM}"

# 3) Create OCR TXT for the full document
python narrative_clinical_pipeline/ocr_paddle_test.py "$PDF"

# 4) Run full document form parser
python checkbox_full_doc_pipeline/parse_full_document_all_pages.py \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle/${DOC_STEM}.txt"

echo ""
echo "Done."
echo "Main outputs:"
echo "  JSON:        /home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/${DOC_STEM}_all_pages_all_specs.json"
echo "  Summary TXT: /home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/${DOC_STEM}_all_pages_all_specs_summary.txt"
echo "  Final TXT:   /home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/${DOC_STEM}_full_final_output.txt"
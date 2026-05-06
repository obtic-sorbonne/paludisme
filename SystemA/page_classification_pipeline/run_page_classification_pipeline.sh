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
echo "Page classification pipeline"
echo "Processing PDF: $PDF"
echo "DOC_STEM: $DOC_STEM"
echo "=============================="

# 1) Create per-page Paddle JSON
python page_classification_pipeline/table_paddle_test.py "$PDF"

# 2) Classify each generated page JSON
for json in /home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/${DOC_STEM}_*_res.json; do
  [ -e "$json" ] || continue

  echo ""
  echo "Classifying: $json"
  python page_classification_pipeline/classify_page_type.py "$json"

  class_json="/home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification/$(basename "$json" .json)_page_type.json"

  echo "Routing: $json"
  python page_classification_pipeline/route_page_processing.py "$json" "$class_json"
done

echo ""
echo "Done."
echo "Main outputs:"
echo "  Paddle page JSONs:   /home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/${DOC_STEM}_*_res.json"
echo "  Classification JSON: /home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification/${DOC_STEM}_*_page_type.json"
echo "  Routing JSON:        saved by route_page_processing.py"
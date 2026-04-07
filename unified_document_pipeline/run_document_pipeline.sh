#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/file.pdf"
  exit 1
fi

PDF="$1"

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "=============================="
echo "Unified document pipeline"
echo "Processing PDF: $PDF"
echo "DOC_STEM: $(basename "$PDF" .pdf)"
echo "=============================="

python unified_document_pipeline/document_pipeline.py "$PDF"

echo ""
echo "Done."
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/folder"
  exit 1
fi

PDF_DIR="$1"

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

for pdf in "$PDF_DIR"/*.pdf; do
  [ -e "$pdf" ] || continue

  echo "=============================="
  echo "Unified document pipeline"
  echo "Processing PDF: $pdf"
  echo "DOC_STEM: $(basename "$pdf" .pdf)"
  echo "=============================="

  python unified_document_pipeline/document_pipeline.py "$pdf"

  echo ""
done

echo "Batch done."
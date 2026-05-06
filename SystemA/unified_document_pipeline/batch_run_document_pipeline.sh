#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /full/path/to/folder"
  exit 1
fi

PDF_DIR="$1"

if [ ! -d "$PDF_DIR" ]; then
  echo "ERROR: Folder not found: $PDF_DIR"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

find "$PDF_DIR" -type f -iname "*.pdf" | sort | while IFS= read -r pdf; do
  doc_stem="$(basename "$pdf" .pdf)"

  echo "=============================="
  echo "Unified document pipeline"
  echo "Processing PDF: $pdf"
  echo "DOC_STEM: $doc_stem"
  echo "=============================="

  # Optional but recommended: remove old outputs for this doc
  rm -rf "benchmark_outputs/final_document_pipeline/$doc_stem"
  rm -f benchmark_outputs/page_routing/${doc_stem}_*_res_route.json
  rm -f benchmark_outputs/page_classification/${doc_stem}_*_page_type.json

  if ! python3 unified_document_pipeline/document_pipeline.py "$pdf"; then
    echo "FAILED: $pdf"
  fi

  echo ""
done

echo "Batch done."
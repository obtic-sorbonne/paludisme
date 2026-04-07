#!/usr/bin/env bash
set -euo pipefail

PDF="${1:-}"

if [ -z "$PDF" ]; then
  echo "Usage: ./narrative_clinical_pipeline/run_narrative_layout_pipeline.sh /full/path/to/file.pdf"
  exit 1
fi

if [ ! -f "$PDF" ]; then
  echo "Error: PDF not found: $PDF"
  exit 1
fi

cd ~/digitize_medical_records
source env_paddle/bin/activate

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PIPE_DIR="/home/lfarooq/digitize_medical_records/narrative_clinical_pipeline"
DOC_STEM="$(basename "$PDF" .pdf)"
OUT_TXT="/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle/${DOC_STEM}.txt"

echo "========================================"
echo "Running narrative clinical pipeline"
echo "PDF:      $PDF"
echo "Doc stem: $DOC_STEM"
echo "========================================"

python "$PIPE_DIR/ocr_paddle_layout_narrative.py" "$PDF"

if [ ! -f "$OUT_TXT" ]; then
  echo "Error: output file not created:"
  echo "  $OUT_TXT"
  exit 1
fi

echo "========================================"
echo "Pipeline finished."
echo "Output:"
echo "  $OUT_TXT"
echo "========================================"
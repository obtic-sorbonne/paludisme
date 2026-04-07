#!/usr/bin/env bash
set -euo pipefail

PDF="${1:-}"
PAGE_INDEX="${2:-0}"

if [ -z "$PDF" ]; then
  echo "Usage: ./lab_tables_pipeline/run_lab_table_pipeline.sh /full/path/to/file.pdf [page_index]"
  exit 1
fi

if [ ! -f "$PDF" ]; then
  echo "Error: PDF not found: $PDF"
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

ROOT_DIR="/home/lfarooq/digitize_medical_records"
LAB_DIR="$ROOT_DIR/experiment/lab_tables_pipeline"
DOC_STEM="$(basename "$PDF" .pdf)"
PAGE_NUM=$((PAGE_INDEX + 1))

PP_JSON="$ROOT_DIR/benchmark_outputs/pp_structure/${DOC_STEM}_${PAGE_INDEX}_res.json"
PADDLE_JSON="$ROOT_DIR/benchmark_outputs/paddle_table/${DOC_STEM}_${PAGE_INDEX}_${PAGE_INDEX}_res.json"
TATR_TXT="$ROOT_DIR/benchmark_outputs/tatr/${DOC_STEM}_p${PAGE_NUM}_detections.txt"
MERGED_TXT="$ROOT_DIR/benchmark_outputs/merged_tables/${DOC_STEM}_${PAGE_INDEX}_res_merged_table.txt"
HYBRID_TXT="$ROOT_DIR/benchmark_outputs/hybrid_tables/${DOC_STEM}_${PAGE_INDEX}_res_hybrid.txt"
FINAL_TXT="$ROOT_DIR/benchmark_outputs/hybrid_tables_final/${DOC_STEM}_${PAGE_INDEX}_res_hybrid_final.txt"

echo "========================================"
echo "Running lab table pipeline"
echo "PDF:        $PDF"
echo "Page index: $PAGE_INDEX"
echo "Page num:   $PAGE_NUM"
echo "Doc stem:   $DOC_STEM"
echo "========================================"

# Reuse PP-Structure output if it already exists
if [ ! -f "$PP_JSON" ]; then
  echo "PP JSON not found for this page. Running PP-Structure on PDF..."
  python "$LAB_DIR/pp_structure_test.py" "$PDF"
else
  echo "Reusing existing PP JSON:"
  echo "  $PP_JSON"
fi

# Reuse per-page Paddle JSON created earlier by page classification pipeline
if [ ! -f "$PADDLE_JSON" ]; then
  echo "Error: Paddle JSON not found: $PADDLE_JSON"
  echo "Expected page-classification pipeline to create it first."
  exit 1
else
  echo "Reusing existing Paddle JSON:"
  echo "  $PADDLE_JSON"
fi

# Run TATR only for the requested page
python "$LAB_DIR/tatr_detect_table.py" "$PDF" --page-index "$PAGE_INDEX"

echo "Using files:"
echo "  PP JSON:      $PP_JSON"
echo "  Paddle JSON:  $PADDLE_JSON"
echo "  TATR TXT:     $TATR_TXT"

if [ ! -f "$PP_JSON" ]; then
  echo "Error: PP JSON not found after PP step: $PP_JSON"
  exit 1
fi

if [ ! -f "$TATR_TXT" ]; then
  echo "Warning: TATR detections file not found: $TATR_TXT"
  echo "Continuing anyway. Downstream parser should handle missing TATR."
fi

python "$LAB_DIR/merge_ppstructure_with_paddle_ocr.py" "$PP_JSON" "$PADDLE_JSON"

if [ ! -f "$MERGED_TXT" ]; then
  echo "Warning: merged output not found after merge step:"
  echo "  $MERGED_TXT"
fi

python "$LAB_DIR/hybrid_table_parser.py" \
  "$PP_JSON" \
  "$PADDLE_JSON" \
  "$TATR_TXT"

if [ ! -f "$HYBRID_TXT" ]; then
  echo "----------------------------------------"
  echo "Hybrid file was not created:"
  echo "  $HYBRID_TXT"
  echo
  echo "This usually means hybrid_table_parser.py could not build a table"
  echo "for this page, often because:"
  echo "  - no usable TATR table was found"
  echo "  - page is mostly narrative and not a true lab-table page"
  echo "  - PP/Paddle/TATR disagreed too much on the table region"
  echo
  echo "Stopping here for this page."
  echo "----------------------------------------"
  exit 1
fi

python "$LAB_DIR/final_hybrid_table_cleaner.py" "$HYBRID_TXT"

if [ ! -f "$FINAL_TXT" ]; then
  echo "Error: final cleaned table file was not created:"
  echo "  $FINAL_TXT"
  exit 1
fi

python "$LAB_DIR/improve_final_table_output.py" "$FINAL_TXT"

echo "========================================"
echo "Lab pipeline finished."
echo "Final file:"
echo "  $FINAL_TXT"
echo "========================================"
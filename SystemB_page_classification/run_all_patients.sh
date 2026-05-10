#!/usr/bin/env bash
# =============================================================
# run_all_patients.sh
# Location: ~/digitize_medical_records/SystemB_page_classification/run_all_patients.sh
#
# Usage:
#   bash run_all_patients.sh                      # all patients
#   bash run_all_patients.sh "2006 RDB 0192"      # one specific patient
#   bash run_all_patients.sh --resume             # skip already done
#   bash run_all_patients.sh --debug              # save debug images
# =============================================================

set -euo pipefail

WORK_DIR="/home/lfarooq/digitize_medical_records"
DATA_DIR="$WORK_DIR/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie"
OUTPUT_BASE="$WORK_DIR/outputs/patients"
PROCESS_SCRIPT="$WORK_DIR/SystemB_page_classification/process_patient.py"
BATCH_REPORT="$OUTPUT_BASE/batch_report.txt"
PYTHON="python"

SPECIFIC_PATIENT=""
RESUME=false
DEBUG_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=true ;;
        --debug)  DEBUG_FLAG="--debug" ;;
        *)        SPECIFIC_PATIENT="$arg" ;;
    esac
done

mkdir -p "$OUTPUT_BASE"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   System B — Batch Patient Processing                   ║"
echo "║   $(date '+%Y-%m-%d %H:%M')                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ -n "$SPECIFIC_PATIENT" ]; then
    PATIENT_FOLDERS=("$DATA_DIR/$SPECIFIC_PATIENT")
    echo "  Mode: Single patient → $SPECIFIC_PATIENT"
else
    mapfile -t PATIENT_FOLDERS < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
    echo "  Mode: All patients"
fi

TOTAL=${#PATIENT_FOLDERS[@]}
echo "  Found: $TOTAL patient folder(s)"
echo "  Resume: $RESUME"
echo ""

{ echo "BATCH REPORT — System B"
  echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Total: $TOTAL"
  echo "============================================================"
} > "$BATCH_REPORT"

PROCESSED=0; SKIPPED=0; FAILED=0

for PATIENT_FOLDER in "${PATIENT_FOLDERS[@]}"; do
    [ -d "$PATIENT_FOLDER" ] || continue

    FOLDER_NAME=$(basename "$PATIENT_FOLDER")
    PATIENT_ID=$(echo "$FOLDER_NAME" | sed 's/^[0-9]\{4\} //' | tr ' ' '_')
    PATIENT_OUTPUT="$OUTPUT_BASE/$PATIENT_ID/${PATIENT_ID}_patient_raw.txt"

    if [ "$RESUME" = true ] && [ -f "$PATIENT_OUTPUT" ]; then
        echo "  ⏭️  Skipping (done): $FOLDER_NAME"
        ((SKIPPED++)) || true
        echo "SKIPPED: $FOLDER_NAME" >> "$BATCH_REPORT"
        continue
    fi

    echo "┌─────────────────────────────────────────────────────────"
    echo "│  [$((PROCESSED+FAILED+SKIPPED+1))/$TOTAL] $FOLDER_NAME"
    echo "└─────────────────────────────────────────────────────────"

    if $PYTHON "$PROCESS_SCRIPT" --folder "$PATIENT_FOLDER" $DEBUG_FLAG; then
        echo "  ✅ Done: $FOLDER_NAME"
        ((PROCESSED++)) || true
        echo "OK: $FOLDER_NAME" >> "$BATCH_REPORT"
    else
        echo "  ❌ Failed: $FOLDER_NAME"
        ((FAILED++)) || true
        echo "FAILED: $FOLDER_NAME" >> "$BATCH_REPORT"
    fi
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   SUMMARY                                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Done    : $PROCESSED"
echo "  Skipped : $SKIPPED"
echo "  Failed  : $FAILED"
echo "  Report  : $BATCH_REPORT"
echo ""

{ echo "============================================================"
  echo "Done: $PROCESSED | Skipped: $SKIPPED | Failed: $FAILED"
  echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$BATCH_REPORT"

[ "$FAILED" -eq 0 ] && echo "  ALL DONE ✅" || { echo "  ⚠️  Some failed"; exit 1; }
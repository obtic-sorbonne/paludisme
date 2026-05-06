#!/usr/bin/env bash
# =============================================================================
# run_patient_pipeline.sh
# System B – Complete patient pipeline: OCR → Classify → Extract → Anonymize → Variables
#
# Location: ~/digitize_medical_records/run_patient_pipeline.sh
#
# Usage:
#   bash run_patient_pipeline.sh --folder "/path/to/2006 RDB 0186/"
#   bash run_patient_pipeline.sh --all
#   bash run_patient_pipeline.sh --folder "/path/..." --skip-ocr
#   bash run_patient_pipeline.sh --all --skip-extraction
#
# Steps:
#   1. GLM-OCR (per PDF)
#   2. Page classification (Qwen 7B vision)
#   3. CNR extraction (Qwen 72B) + GLM lab text
#   4. Merge all docs → patient_raw.txt
#   5. Anonymize → patient_anonymized.txt + replacements.csv
#   6. Variable extraction → research_table.xlsx + research_database.db
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR="$HOME/digitize_medical_records"
PYTHON="$WORK_DIR/labelimg_env/bin/python"
DATA_DIR="$WORK_DIR/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie"
FINAL_ANON="$WORK_DIR/outputs/final_anonymized"

PROCESS_PATIENT="$WORK_DIR/SystemB_page_classification/process_patient.py"
EXTRACT_VARS="$WORK_DIR/VariableExtraction/extract_variables.py"
VAR_CONFIG="$WORK_DIR/VariableExtraction/variable_extraction_config.yaml"
OUTPUT_DIR="$WORK_DIR/VariableExtraction/outputs"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

banner() {
    echo -e "\n${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║  $1${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
}

ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "  ${RED}❌ $1${NC}"; }
step() { echo -e "\n  ${BOLD}── $1${NC}"; }

# ── Arg parsing ───────────────────────────────────────────────────────────────
FOLDER=""
RUN_ALL=false
SKIP_OCR=false
SKIP_EXTRACTION=false
SKIP_VARIABLES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folder)        FOLDER="$2"; shift 2 ;;
        --all)           RUN_ALL=true; shift ;;
        --skip-ocr)      SKIP_OCR=true; shift ;;
        --skip-extraction) SKIP_EXTRACTION=true; shift ;;
        --skip-variables)  SKIP_VARIABLES=true; shift ;;
        -h|--help)
            echo "Usage: bash run_patient_pipeline.sh --folder <path> | --all [options]"
            echo "Options:"
            echo "  --skip-ocr          Skip GLM-OCR (if already done)"
            echo "  --skip-extraction   Skip steps 1-4 (run anonymization + variables only)"
            echo "  --skip-variables    Skip variable extraction step"
            exit 0 ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$RUN_ALL" == false && -z "$FOLDER" ]]; then
    err "Provide --folder <path> or --all"
    exit 1
fi

# ── Collect patient folders ───────────────────────────────────────────────────
if [[ "$RUN_ALL" == true ]]; then
    mapfile -t FOLDERS < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
    if [[ ${#FOLDERS[@]} -eq 0 ]]; then
        err "No folders found in: $DATA_DIR"
        exit 1
    fi
    echo -e "${BOLD}Found ${#FOLDERS[@]} patient folder(s)${NC}"
else
    if [[ ! -d "$FOLDER" ]]; then
        err "Folder not found: $FOLDER"
        exit 1
    fi
    FOLDERS=("$FOLDER")
fi

# ── Track results ─────────────────────────────────────────────────────────────
TOTAL=${#FOLDERS[@]}
DONE=0
FAILED=0
FAILED_LIST=()

START_TIME=$SECONDS

# ── Process each patient ──────────────────────────────────────────────────────
for folder in "${FOLDERS[@]}"; do
    PATIENT_NAME="$(basename "$folder")"
    banner "PATIENT: $PATIENT_NAME ($((DONE+FAILED+1))/$TOTAL)"

    # ── Steps 1-5: OCR + classify + extract + merge + anonymize ──────────────
    if [[ "$SKIP_EXTRACTION" == false ]]; then
        step "Steps 1-5: OCR → Classify → Extract → Merge → Anonymize"
        if "$PYTHON" "$PROCESS_PATIENT" --folder "$folder"; then
            ok "Steps 1-5 complete"
        else
            err "Steps 1-5 FAILED for: $PATIENT_NAME"
            FAILED=$((FAILED+1))
            FAILED_LIST+=("$PATIENT_NAME")
            continue
        fi
    else
        warn "Skipping steps 1-5 (--skip-extraction)"
    fi

    # ── Step 6: Variable extraction ───────────────────────────────────────────
    if [[ "$SKIP_VARIABLES" == false ]]; then
        step "Step 6: Variable extraction → Excel + SQLite"

        # Find the anonymized file for this patient
        # Patient ID comes from the folder name, e.g. "2006 RDB 0186" → RDB_0186
        PATIENT_ID=$(echo "$PATIENT_NAME" | grep -oP 'RDB_\d+' || echo "")
        if [[ -z "$PATIENT_ID" ]]; then
            # Try extracting numeric ID from folder name
            PATIENT_ID=$(echo "$PATIENT_NAME" | grep -oP '\d{4}' | tail -1 || echo "")
        fi

        # Find the anonymized file (it uses a sequential number, not the folder name)
        # Find the most recently written anonymized file matching this patient
        ANON_FILE=$(find "$FINAL_ANON" -name "patient_*_anonymized.txt" \
                    -newer "$WORK_DIR/outputs" 2>/dev/null | sort | tail -1 || true)

        # Fall back: find any anonymized file written in the last 10 minutes
        if [[ -z "$ANON_FILE" ]]; then
            ANON_FILE=$(find "$FINAL_ANON" -name "patient_*_anonymized.txt" \
                        -mmin -10 2>/dev/null | sort | tail -1 || true)
        fi

        if [[ -n "$ANON_FILE" ]]; then
            if "$PYTHON" "$EXTRACT_VARS" \
                --patient "$ANON_FILE" \
                --config "$VAR_CONFIG" \
                --output-dir "$OUTPUT_DIR"; then
                ok "Variable extraction complete"
            else
                warn "Variable extraction failed (pipeline still counted as done)"
            fi
        else
            warn "No anonymized file found for this patient - skipping variable extraction"
        fi
    else
        warn "Skipping variable extraction (--skip-variables)"
    fi

    DONE=$((DONE+1))
    ok "Patient complete: $PATIENT_NAME"
done

# ── Summary ───────────────────────────────────────────────────────────────────
ELAPSED=$((SECONDS - START_TIME))
MINS=$((ELAPSED/60)); SECS=$((ELAPSED%60))

banner "PIPELINE COMPLETE"
echo -e "  ${GREEN}✅ Success: $DONE / $TOTAL${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}❌ Failed:  $FAILED / $TOTAL${NC}"
    for f in "${FAILED_LIST[@]}"; do
        echo -e "     - $f"
    done
fi
echo -e "  ⏱️  Time: ${MINS}m ${SECS}s"
echo ""
echo -e "  📊 Outputs:"
echo -e "     Anonymized files: $FINAL_ANON/"
echo -e "     Excel table:      $OUTPUT_DIR/research_table.xlsx"
echo -e "     SQLite database:  $OUTPUT_DIR/research_database.db"
echo ""
echo -e "  🔍 Query the database:"
echo -e "     sqlite3 $OUTPUT_DIR/research_database.db \\"
echo -e "       'SELECT ID_Patient, Sexe, Age, gravite_palu, hemoglobine_J0 FROM patients;'"
echo ""

exit $( [[ $FAILED -eq 0 ]] && echo 0 || echo 1 )
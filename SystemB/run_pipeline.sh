#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  –  System B Complete Pipeline
# Location: ~/digitize_medical_records/run_pipeline.sh
#
# Smart auto-detection: pass ONE path and it figures out what to do:
#
#   1. Path to a single PDF file     → process that one PDF only
#   2. Path to a patient folder      → process all PDFs in that folder
#   3. Path to a folder of folders   → process all patient subfolders
#
# Usage:
#   bash run_pipeline.sh "/path/to/DOC_00118.pdf"
#   bash run_pipeline.sh "/path/to/2006 RDB 0186/"
#   bash run_pipeline.sh "/path/to/Batch de docs/"
#
# Optional flags (add after the path):
#   --skip-extraction   Skip OCR/classify/extract/anonymize (re-run variables only)
#   --skip-variables    Skip Excel/SQLite step
#   --help              Show this help
# =============================================================================

set -euo pipefail

# ── Paths (change these if your layout changes) ────────────────────────────────
WORK_DIR="$HOME/digitize_medical_records"
PYTHON="$WORK_DIR/labelimg_env/bin/python"
PROCESS_PATIENT="$WORK_DIR/SystemB_page_classification/process_patient.py"
EXTRACT_VARS="$WORK_DIR/VariableExtraction/extract_variables.py"
VAR_CONFIG="$WORK_DIR/VariableExtraction/variable_extraction_config.yaml"
VAR_OUTPUT="$WORK_DIR/VariableExtraction/outputs"
FINAL_ANON="$WORK_DIR/outputs/final_anonymized"

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

banner() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; 
           echo -e "${BOLD}${BLUE}  $1${NC}";
           echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }
ok()     { echo -e "  ${GREEN}✅ $1${NC}"; }
warn()   { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
err()    { echo -e "  ${RED}❌ $1${NC}"; }
step()   { echo -e "\n  ${BOLD}▶ $1${NC}"; }

# ── Args ───────────────────────────────────────────────────────────────────────
INPUT_PATH=""
SKIP_EXTRACTION=false
SKIP_VARIABLES=false

if [[ $# -eq 0 ]]; then
    err "No path provided."
    echo "Usage: bash run_pipeline.sh <path> [--skip-extraction] [--skip-variables]"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --skip-extraction) SKIP_EXTRACTION=true ;;
        --skip-variables)  SKIP_VARIABLES=true ;;
        --help|-h)
            echo "Usage: bash run_pipeline.sh <path> [options]"
            echo ""
            echo "  <path> can be:"
            echo "    A single PDF file     → process that PDF"
            echo "    A patient folder      → process all PDFs in folder"
            echo "    A folder of folders   → process all patient subfolders"
            echo ""
            echo "Options:"
            echo "  --skip-extraction   Skip OCR/classify/anonymize (variables only)"
            echo "  --skip-variables    Skip Excel/SQLite output"
            exit 0 ;;
        -*) err "Unknown option: $arg"; exit 1 ;;
        *)  INPUT_PATH="$arg" ;;
    esac
done

if [[ -z "$INPUT_PATH" ]]; then
    err "No path provided."
    exit 1
fi

if [[ ! -e "$INPUT_PATH" ]]; then
    err "Path not found: $INPUT_PATH"
    exit 1
fi

# ── Detect input type and collect patient folders ──────────────────────────────
declare -a PATIENT_FOLDERS

if [[ -f "$INPUT_PATH" ]]; then
    # ── Case 1: Single PDF file ────────────────────────────────────────────────
    if [[ "${INPUT_PATH,,}" != *.pdf ]]; then
        err "File is not a PDF: $INPUT_PATH"
        exit 1
    fi
    # The patient folder is the parent directory of the PDF
    PATIENT_FOLDERS=("$(dirname "$INPUT_PATH")")
    echo -e "${BOLD}Mode: Single PDF file${NC}"
    echo -e "  File:   $INPUT_PATH"
    echo -e "  Folder: ${PATIENT_FOLDERS[0]}"

elif [[ -d "$INPUT_PATH" ]]; then
    # Count direct PDF children
    PDF_COUNT=$(find "$INPUT_PATH" -maxdepth 1 \( -name "*.pdf" -o -name "*.PDF" \) | wc -l)
    # Count subfolders that contain PDFs
    SUBFOLDER_COUNT=0
    while IFS= read -r -d '' d; do
        if find "$d" -maxdepth 1 \( -name "*.pdf" -o -name "*.PDF" \) | grep -q .; then
            SUBFOLDER_COUNT=$((SUBFOLDER_COUNT+1))
        fi
    done < <(find "$INPUT_PATH" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

    if [[ $PDF_COUNT -gt 0 && $SUBFOLDER_COUNT -eq 0 ]]; then
        # ── Case 2: Patient folder (contains PDFs directly) ───────────────────
        PATIENT_FOLDERS=("$INPUT_PATH")
        echo -e "${BOLD}Mode: Single patient folder${NC}"
        echo -e "  Folder: $INPUT_PATH"
        echo -e "  PDFs:   $PDF_COUNT file(s)"

    elif [[ $SUBFOLDER_COUNT -gt 0 ]]; then
        # ── Case 3: Folder of patient folders ─────────────────────────────────
        while IFS= read -r -d '' d; do
            if find "$d" -maxdepth 1 \( -name "*.pdf" -o -name "*.PDF" \) | grep -q .; then
                PATIENT_FOLDERS+=("$d")
            fi
        done < <(find "$INPUT_PATH" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
        echo -e "${BOLD}Mode: Folder of patient folders${NC}"
        echo -e "  Parent:   $INPUT_PATH"
        echo -e "  Patients: ${#PATIENT_FOLDERS[@]} folder(s) found"
    else
        err "No PDFs found in: $INPUT_PATH"
        exit 1
    fi
fi

TOTAL=${#PATIENT_FOLDERS[@]}
if [[ $TOTAL -eq 0 ]]; then
    err "No patient folders found."
    exit 1
fi

# ── Run pipeline ───────────────────────────────────────────────────────────────
DONE=0; FAILED=0
declare -a FAILED_LIST
START_TIME=$SECONDS

for folder in "${PATIENT_FOLDERS[@]}"; do
    PATIENT_NAME="$(basename "$folder")"
    banner "[$((DONE+FAILED+1))/$TOTAL] $PATIENT_NAME"

    # ── Steps 1-5 ─────────────────────────────────────────────────────────────
    if [[ "$SKIP_EXTRACTION" == false ]]; then
        step "Step 1-5: GLM-OCR → Classify → Extract → Merge → Anonymize"
        BEFORE_TS=$(date +%s)
        sleep 1
        if "$PYTHON" "$PROCESS_PATIENT" --folder "$folder"; then
            ok "Steps 1-5 complete"
        else
            err "FAILED: $PATIENT_NAME"
            FAILED=$((FAILED+1)); FAILED_LIST+=("$PATIENT_NAME")
            continue
        fi
    else
        warn "Skipping steps 1-5 (--skip-extraction)"
        BEFORE_TS=0
    fi

    DONE=$((DONE+1))
    ok "Done: $PATIENT_NAME"
done

# ── Step 6: Variable extraction (runs ONCE at end for ALL patients) ───────────
# Running at the end guarantees every patient is in the same Excel/SQLite,
# even if some patients were processed in previous runs.
if [[ "$SKIP_VARIABLES" == false ]]; then
    banner "Step 6: Variable extraction → Excel + SQLite (all patients)"
    mkdir -p "$VAR_OUTPUT"
    if "$PYTHON" "$EXTRACT_VARS" \
        --all \
        --config "$VAR_CONFIG" \
        --output-dir "$VAR_OUTPUT"; then
        ok "All patients written to research_table.xlsx + research_database.db"
    else
        warn "Variable extraction had errors - check output above"
    fi
else
    warn "Skipping variable extraction (--skip-variables)"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
ELAPSED=$((SECONDS - START_TIME))
banner "DONE"
echo -e "  ${GREEN}✅ Success : $DONE / $TOTAL${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}❌ Failed  : $FAILED / $TOTAL${NC}"
    for f in "${FAILED_LIST[@]}"; do echo -e "     - $f"; done
fi
echo -e "  ⏱️  Time   : $((ELAPSED/60))m $((ELAPSED%60))s"
echo ""
echo -e "  📁 Results saved to:"
echo -e "     $VAR_OUTPUT/research_table.xlsx"
echo -e "     $VAR_OUTPUT/research_database.db"
echo ""
exit $([[ $FAILED -eq 0 ]] && echo 0 || echo 1)
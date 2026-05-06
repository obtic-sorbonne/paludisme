#!/usr/bin/env bash
# =============================================================
# run_pipeline.sh
#
# Full pipeline for CNR Paludisme form digitization:
#   1. Run GLM-OCR on each PDF (page by page)
#   2. Run Qwen2.5-VL 72B extraction
#   3. Run postprocess checker (GLM cross-check)
#   4. Run evaluation against gold sets
#
# Usage:
#   bash run_pipeline.sh              # process all docs
#   bash run_pipeline.sh DOC_00116    # process one doc only
# =============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
BASE_DATA="/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie"
WORK_DIR="/home/lfarooq/digitize_medical_records"
PRED_DIR="$WORK_DIR/evaluation/predictions"
GOLD_DIR="$WORK_DIR/evaluation/gold"
GLM_BASE="$HOME/glm_ocr_runs"
GLM_MODEL="glm-ocr:latest"
QWEN_MODEL="qwen2.5vl:72b"

# ── Document registry (doc_id → pdf path) ────────────────────
declare -A DOCS
DOCS["DOC_00005"]="$BASE_DATA/2006 RDB 0202/DOC_00005.pdf"
DOCS["DOC_00098"]="$BASE_DATA/2006 RDB 0156/DOC_00098.pdf"
DOCS["DOC_00116"]="$BASE_DATA/2006 RDB 0185/DOC_00116.pdf"
DOCS["DOC_00173"]="$BASE_DATA/2006 RDB 0198/DOC_00173.pdf"
DOCS["DOC_00184"]="$BASE_DATA/2006 RDB 0015/DOC_00184.pdf"
DOCS["DOC_00192"]="$BASE_DATA/2006 RDB 0201/DOC_00192.pdf"
DOCS["DOC_00110"]="$BASE_DATA/2006 RDB 0185/DOC_00110.pdf"

# ── Which docs to process ─────────────────────────────────────
if [ "$#" -ge 1 ]; then
    # Single doc mode
    TARGET_DOCS=("$1")
else
    # All docs
    TARGET_DOCS=("DOC_00005" "DOC_00098" "DOC_00116" "DOC_00173" "DOC_00184" "DOC_00192")
fi

# ── Helper: GLM-OCR one PDF ───────────────────────────────────
run_glm() {
    local DOC_ID="$1"
    local PDF="$2"
    local WORKDIR="$GLM_BASE/$DOC_ID"
    local IMGDIR="$WORKDIR/pages"
    local TXTDIR="$WORKDIR/text"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [GLM-OCR] $DOC_ID"
    echo "══════════════════════════════════════════════════════"

    # Skip if already done
    if [ -d "$TXTDIR" ] && [ "$(ls -A $TXTDIR 2>/dev/null)" ]; then
        echo "  ✓ GLM text already exists — skipping"
        return 0
    fi

    mkdir -p "$IMGDIR" "$TXTDIR"

    echo "  [1/3] Converting PDF to PNG pages..."
    pdftoppm -png "$PDF" "$IMGDIR/page" >/dev/null 2>&1
    PAGE_COUNT=$(find "$IMGDIR" -maxdepth 1 -type f -name 'page-*.png' | wc -l)
    echo "  Pages found: $PAGE_COUNT"

    if [ "$PAGE_COUNT" -eq 0 ]; then
        echo "  ERROR: No pages created from PDF"
        return 1
    fi

    echo "  [2/3] Running GLM-OCR page by page..."
    local MERGED="$WORKDIR/${DOC_ID}_full_ocr.txt"
    : > "$MERGED"

    for IMG in "$IMGDIR"/page-*.png; do
        PAGE_NAME="$(basename "$IMG" .png)"
        OUT_TXT="$TXTDIR/${PAGE_NAME}.txt"
        echo "    Processing: $PAGE_NAME"
        ollama run "$GLM_MODEL" "$IMG
Extract the text in the image exactly. Preserve line breaks as much as possible." > "$OUT_TXT"
        echo "===== ${PAGE_NAME} =====" >> "$MERGED"
        cat "$OUT_TXT" >> "$MERGED"
        echo "" >> "$MERGED"
    done

    echo "  [3/3] GLM-OCR done → $TXTDIR"
}

# ── Helper: Qwen extraction ───────────────────────────────────
run_qwen() {
    local DOC_ID="$1"
    local PDF="$2"
    local OUT="$PRED_DIR/${DOC_ID}.txt"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [QWEN] $DOC_ID"
    echo "══════════════════════════════════════════════════════"

    python "$WORK_DIR/extract_form_qwen.py" \
        --pdf "$PDF" \
        --out "$OUT" \
        --model "$QWEN_MODEL"

    echo "  ✓ Qwen extraction done → $OUT"
}

# ── Helper: Postprocess checker ───────────────────────────────
run_checker() {
    local DOC_ID="$1"
    local PRED="$PRED_DIR/${DOC_ID}.txt"
    local GLM_DIR="$GLM_BASE/$DOC_ID/text"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [CHECKER] $DOC_ID"
    echo "══════════════════════════════════════════════════════"

    if [ ! -d "$GLM_DIR" ]; then
        echo "  ⚠ No GLM text found — skipping checker"
        return 0
    fi

    python "$WORK_DIR/postprocess_checker.py" \
        --pred "$PRED" \
        --glm_dir "$GLM_DIR" \
        --inplace

    echo "  ✓ Checker done"
}

# ── Main loop ─────────────────────────────────────────────────
mkdir -p "$PRED_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   CNR Paludisme Pipeline — $(date '+%Y-%m-%d %H:%M')          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Processing: ${TARGET_DOCS[*]}"
echo ""

for DOC_ID in "${TARGET_DOCS[@]}"; do
    if [ -z "${DOCS[$DOC_ID]+_}" ]; then
        echo "ERROR: Unknown doc ID: $DOC_ID"
        echo "Available: ${!DOCS[*]}"
        exit 1
    fi

    PDF="${DOCS[$DOC_ID]}"

    if [ ! -f "$PDF" ]; then
        echo "ERROR: PDF not found: $PDF"
        exit 1
    fi

    echo ""
    echo "┌─────────────────────────────────────────────────────"
    echo "│  Processing: $DOC_ID"
    echo "└─────────────────────────────────────────────────────"

    # Step 1: GLM-OCR
    run_glm "$DOC_ID" "$PDF"

    # Step 2: Qwen extraction
    run_qwen "$DOC_ID" "$PDF"

    # Step 3: Post-process checker
    run_checker "$DOC_ID"

    echo ""
    echo "  ✅ $DOC_ID complete"
done

# ── Final evaluation ──────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Running final evaluation...                        ║"
echo "╚══════════════════════════════════════════════════════╝"

REPORT="$WORK_DIR/evaluation/report_final.txt"
python "$WORK_DIR/evaluation/compare.py" \
    --gold_dir "$GOLD_DIR" \
    --pred_dir "$PRED_DIR" \
    --out "$REPORT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ALL DONE                                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Report saved → $REPORT"
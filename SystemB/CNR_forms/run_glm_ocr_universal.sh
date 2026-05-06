#!/usr/bin/env bash
# =============================================================
# run_glm_ocr_universal.sh
# Location: ~/digitize_medical_records/CNR_forms/run_glm_ocr_universal.sh
#
# Universal GLM-OCR - accepts ANY PDF path, no hardcoded registry.
#
# Usage:
#   bash run_glm_ocr_universal.sh "/path/to/any/DOC_00185.pdf"
#
# Output:
#   ~/glm_ocr_runs/<DOC_NAME>/
#     ├── pages/page-01.png ...   (page images)
#     ├── text/page-01.txt  ...   (per-page OCR text)
#     └── <DOC_NAME>_full_ocr.txt (merged full text)
# =============================================================

set -euo pipefail

# ── Validate input ─────────────────────────────────────────────
if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_glm_ocr_universal.sh /path/to/file.pdf"
    exit 1
fi

PDF="$1"

if [ ! -f "$PDF" ]; then
    echo "ERROR: PDF not found: $PDF"
    exit 1
fi

# ── Config ─────────────────────────────────────────────────────
GLM_MODEL="glm-ocr:latest"
GLM_BASE="$HOME/glm_ocr_runs"

# Extract doc ID from filename (e.g. DOC_00185.pdf → DOC_00185)
DOC_ID="$(basename "$PDF" .pdf)"

WORKDIR="$GLM_BASE/$DOC_ID"
IMGDIR="$WORKDIR/pages"
TXTDIR="$WORKDIR/text"
MERGED="$WORKDIR/${DOC_ID}_full_ocr.txt"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   GLM-OCR — $(date '+%Y-%m-%d %H:%M')                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  PDF    : $PDF"
echo "  Doc ID : $DOC_ID"
echo "  Output : $WORKDIR"

# ── Skip if already done ───────────────────────────────────────
if [ -d "$TXTDIR" ] && [ "$(ls -A "$TXTDIR" 2>/dev/null)" ]; then
    echo "  ✅ Already done — skipping (delete $TXTDIR to rerun)"
    exit 0
fi

mkdir -p "$IMGDIR" "$TXTDIR"

# ── Step 1: PDF → PNG pages ────────────────────────────────────
echo ""
echo "  [1/3] Converting PDF to PNG pages..."
pdftoppm -png -r 150 "$PDF" "$IMGDIR/page" 2>/dev/null
PAGE_COUNT=$(find "$IMGDIR" -maxdepth 1 -type f -name 'page-*.png' | wc -l)
echo "  Pages: $PAGE_COUNT"

if [ "$PAGE_COUNT" -eq 0 ]; then
    echo "  ❌ No pages created from PDF"
    exit 1
fi

# ── Step 2: GLM-OCR each page ─────────────────────────────────
echo ""
echo "  [2/3] Running GLM-OCR page by page..."
: > "$MERGED"

for IMG in $(find "$IMGDIR" -maxdepth 1 -name 'page-*.png' | sort); do
    PAGE_NAME="$(basename "$IMG" .png)"
    OUT_TXT="$TXTDIR/${PAGE_NAME}.txt"

    echo "    → $PAGE_NAME"
    ollama run "$GLM_MODEL" "$IMG
Extract the text in the image exactly. Preserve line breaks as much as possible." > "$OUT_TXT"

    echo "===== ${PAGE_NAME} =====" >> "$MERGED"
    cat "$OUT_TXT" >> "$MERGED"
    echo "" >> "$MERGED"
done

# ── Step 3: Done ───────────────────────────────────────────────
echo ""
echo "  [3/3] Done"
echo "  Text files : $TXTDIR"
echo "  Full OCR   : $MERGED"
echo ""
echo "  ✅ GLM-OCR complete: $DOC_ID"
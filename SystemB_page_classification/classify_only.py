#!/usr/bin/env python3
"""
classify_only.py - DRY RUN ONLY, no Qwen, no GLM-OCR pipeline
"""
import sys, argparse
from pathlib import Path

WORK_DIR     = "/home/lfarooq/digitize_medical_records"
GLM_OCR_BASE = "/home/lfarooq/glm_ocr_runs"

sys.path.insert(0, f"{WORK_DIR}/SystemB_page_classification")
from classify_page import is_cnr_form

def get_or_extract_pages(pdf_path):
    doc_id    = pdf_path.stem
    pages_dir = Path(GLM_OCR_BASE) / doc_id / "pages"
    if pages_dir.exists() and any(
        f for f in pages_dir.iterdir()
        if f.suffix.lower() in {".png",".jpg",".jpeg"}
        and "page" in f.name.lower()
        and "_classified" not in f.name
    ):
        print(f"  Using existing GLM pages: {pages_dir}")
        return pages_dir
    print(f"  Extracting pages from {doc_id}...")
    try:
        import fitz
        pages_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            pix.save(str(pages_dir / f"page-{i+1:02d}.png"))
        total = len(doc)
        doc.close()
        print(f"  Extracted {total} pages")
        return pages_dir
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def classify_pdf(pdf_path, debug=False):
    doc_id = pdf_path.stem
    print(f"\n  {'─'*52}")
    print(f"  PDF: {doc_id}")
    print(f"  {'─'*52}")
    pages_dir = get_or_extract_pages(pdf_path)
    if not pages_dir:
        return {"doc_id": doc_id, "error": True, "cnr_pages": [], "non_cnr_pages": []}

    exts = {".png", ".jpg", ".jpeg"}
    page_files = sorted(
        f for f in pages_dir.iterdir()
        if f.suffix.lower() in exts
        and "page" in f.name.lower()
        and "_classified" not in f.name
    )
    print(f"  Found {len(page_files)} pages")
    cnr_pages, non_cnr_pages = [], []
    for page_file in page_files:
        try:
            page_num = int(page_file.stem.split("-")[-1])
        except ValueError:
            page_num = len(cnr_pages) + len(non_cnr_pages) + 1
        result = is_cnr_form(str(page_file), min_radio_groups=3, debug=debug)
        label  = "✅ CNR    " if result["is_cnr"] else "📄 non-CNR"
        print(f"    Page {page_num:2d}: {label}  ({result['details']})")
        if result["is_cnr"]:
            cnr_pages.append(page_num)
        else:
            non_cnr_pages.append(page_num)

    pipeline = "→ Qwen + Checker" if cnr_pages else "→ GLM-OCR only"
    print(f"\n  CNR pages     : {cnr_pages}")
    print(f"  Non-CNR pages : {non_cnr_pages}")
    print(f"  Pipeline      : {pipeline}")
    return {"doc_id": doc_id, "cnr_pages": cnr_pages, "non_cnr_pages": non_cnr_pages}

def main():
    parser = argparse.ArgumentParser(description="Classify pages — NO pipeline runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",    help="Path to one PDF")
    group.add_argument("--folder", help="Path to patient folder")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Classification Only — no Qwen, no GLM-OCR")
    print(f"{'='*55}")

    if args.pdf:
        results = [classify_pdf(Path(args.pdf), debug=args.debug)]
    else:
        folder = Path(args.folder)
        pdfs = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".pdf")
        if not pdfs:
            print(f"No PDFs in {folder}"); sys.exit(1)
        print(f"  Folder: {folder.name} — {len(pdfs)} PDFs")
        results = [classify_pdf(p, debug=args.debug) for p in pdfs]

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    cnr_docs    = [r for r in results if r.get("cnr_pages")]
    noncnr_docs = [r for r in results if not r.get("cnr_pages") and not r.get("error")]
    print(f"  CNR docs ({len(cnr_docs)}) → Qwen + Checker:")
    for r in cnr_docs:
        print(f"    {r['doc_id']}  CNR pages: {r['cnr_pages']}")
    print(f"  Non-CNR docs ({len(noncnr_docs)}) → GLM only:")
    for r in noncnr_docs:
        print(f"    {r['doc_id']}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()

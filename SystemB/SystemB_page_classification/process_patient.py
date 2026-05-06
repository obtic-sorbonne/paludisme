import re
#!/usr/bin/env python3
"""
process_patient.py
Location: ~/digitize_medical_records/SystemB_page_classification/process_patient.py

For each PDF:
  1. GLM-OCR (text + page images)
  2. Classify each page (CNR or non-CNR) using Qwen 7B
  3a. CNR pages  → Qwen 72B extraction + postprocess checker
  3b. Non-CNR pages → GLM-OCR text for those specific pages
  3c. Mixed docs → combine both outputs into one file
  4. Merge all doc outputs → one patient_raw.txt
"""

import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR        = "/home/lfarooq/digitize_medical_records"
PYTHON          = f"{WORK_DIR}/labelimg_env/bin/python"
GLM_SCRIPT      = f"{WORK_DIR}/CNR_forms/run_glm_ocr_universal.sh"
GLM_OCR_BASE    = "/home/lfarooq/glm_ocr_runs"
CNR_DIR         = f"{WORK_DIR}/CNR_forms"
QWEN_SCRIPT     = f"{CNR_DIR}/extract_form_qwen.py"
CHECKER_SCRIPT  = f"{CNR_DIR}/postprocess_checker.py"
OUTPUT_BASE     = f"{WORK_DIR}/outputs/patients"
QWEN_MODEL      = "qwen72b-limited"

sys.path.insert(0, f"{WORK_DIR}/SystemB_page_classification")
sys.path.insert(0, f"{WORK_DIR}/Anonymization_systemB")
from classify_page import is_cnr_form
from fix_glm_tables import fix_file as fix_glm_tables

# Anonymization config
ANON_SCRIPT  = f"{WORK_DIR}/Anonymization_systemB/anonymize_patients.py"
ANON_CONFIG  = f"{WORK_DIR}/Anonymization_systemB/anonymization_config.yaml"
FINAL_OUTPUT = f"{WORK_DIR}/outputs/final_anonymized"  # all anonymized files go here

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "  ✅", "WARN": "  ⚠️ ",
             "ERROR": "  ❌", "STEP": "\n  ──"}
    print(f"{icons.get(level, '  ')} {msg}", flush=True)


def run_cmd(cmd, description="", timeout=1800):
    if description:
        log(description)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


# ── Patient ID ────────────────────────────────────────────────────────────────
def get_patient_id(folder_path: Path) -> str:
    parts = folder_path.name.strip().split()
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        parts = parts[1:]
    return "_".join(parts)


# ── Step 1: GLM-OCR ───────────────────────────────────────────────────────────
def run_glm_ocr(pdf_path: Path) -> dict:
    doc_id    = pdf_path.stem
    glm_dir   = Path(GLM_OCR_BASE) / doc_id
    text_dir  = glm_dir / "text"
    pages_dir = glm_dir / "pages"
    full_ocr  = glm_dir / f"{doc_id}_full_ocr.txt"

    if text_dir.exists() and any(text_dir.iterdir()):
        log(f"GLM-OCR already done for {doc_id} — skipping", "OK")
        return {"doc_id": doc_id, "glm_dir": str(glm_dir),
                "text_dir": str(text_dir), "pages_dir": str(pages_dir),
                "full_ocr": str(full_ocr), "success": True}

    log(f"Running GLM-OCR on {doc_id}...")
    ok, out = run_cmd(f'bash "{GLM_SCRIPT}" "{pdf_path}"', timeout=1800)
    if not ok:
        log(f"GLM-OCR failed: {out}", "ERROR")
        return {"doc_id": doc_id, "success": False}

    log(f"GLM-OCR done → {glm_dir}", "OK")

    # Fix known table parsing issues in GLM text files
    txt_files = sorted(Path(text_dir).glob("*.txt")) if Path(text_dir).exists() else []
    fixed_count = sum(1 for f in txt_files if fix_glm_tables(f, inplace=True))
    if fixed_count:
        log(f"Fixed table labels in {fixed_count} page(s)", "OK")

    return {"doc_id": doc_id, "glm_dir": str(glm_dir),
            "text_dir": str(text_dir), "pages_dir": str(pages_dir),
            "full_ocr": str(full_ocr), "success": True}


# ── Step 2: Classify pages ────────────────────────────────────────────────────
def classify_pages(pages_dir: str, debug: bool = False) -> dict:
    pages_dir = Path(pages_dir)
    if not pages_dir.exists():
        log(f"Pages dir not found: {pages_dir}", "WARN")
        return {}

    exts = {".png", ".jpg", ".jpeg"}
    page_files = sorted(
        f for f in pages_dir.iterdir()
        if f.suffix.lower() in exts
        and "page" in f.name.lower()
        and "_classified" not in f.name
    )

    # KEY FIX: deduplicate pages by number
    # page-1.png and page-01.png are the same page - keep only one
    seen_nums = set()
    deduped_files = []
    for page_file in page_files:
        try:
            page_num = int(page_file.stem.split("-")[-1])
        except ValueError:
            page_num = len(deduped_files) + 1
        if page_num not in seen_nums:
            seen_nums.add(page_num)
            deduped_files.append((page_num, page_file))

    results = {}
    for page_num, page_file in deduped_files:
        result = is_cnr_form(str(page_file), debug=debug)
        result["page_file"] = str(page_file)
        results[page_num] = result
        label = "✅ CNR" if result["is_cnr"] else "📄 non-CNR"
        log(f"  Page {page_num}: {label}  ({result['details']})")

    return results


# ── Step 3a: CNR pipeline (Qwen on CNR pages only) ───────────────────────────
def run_qwen_on_cnr_pages(glm_info: dict, page_results: dict,
                           output_dir: Path) -> str:
    """Run Qwen+Checker on CNR pages only. Returns path to output txt."""
    doc_id = glm_info["doc_id"]
    out_path = output_dir / f"{doc_id}_cnr_output.txt"

    cnr_pages = {num: info for num, info in page_results.items()
                 if info["is_cnr"]}
    if not cnr_pages:
        return None

    log(f"CNR pages to process with Qwen: {sorted(cnr_pages.keys())}")

    # Copy only CNR pages to temp dir
    cnr_img_dir = output_dir / f"{doc_id}_cnr_pages"
    cnr_img_dir.mkdir(exist_ok=True)

    for i, (page_num, info) in enumerate(sorted(cnr_pages.items()), 1):
        src = Path(info["page_file"])
        dst = cnr_img_dir / f"page-{i:02d}{src.suffix}"
        shutil.copy(str(src), str(dst))
        log(f"  Copied CNR page {page_num} → {dst.name}")

    # Run Qwen
    cmd = (f'"{PYTHON}" "{QWEN_SCRIPT}" '
           f'--img_dir "{cnr_img_dir}" '
           f'--glm_dir "{glm_info["text_dir"]}" '
           f'--out "{out_path}" '
           f'--model "{QWEN_MODEL}"')
    ok, out = run_cmd(cmd, f"Qwen on {len(cnr_pages)} CNR pages...", timeout=3600)

    # Clean up temp dir
    shutil.rmtree(str(cnr_img_dir), ignore_errors=True)

    if not ok:
        log(f"Qwen failed: {out}", "ERROR")
        return None

    # Run checker
    cmd = (f'"{PYTHON}" "{CHECKER_SCRIPT}" '
           f'--pred "{out_path}" '
           f'--glm_dir "{glm_info["text_dir"]}" '
           f'--inplace')
    ok, out = run_cmd(cmd, "Running checker...", timeout=300)
    if not ok:
        log(f"Checker warning: {out}", "WARN")

    log(f"Qwen + Checker done → {out_path}", "OK")
    return str(out_path)


# ── Step 3b: Extract non-CNR pages text from GLM ─────────────────────────────
def extract_non_cnr_text(glm_info: dict, page_results: dict,
                          output_dir: Path) -> str:
    """
    Extract GLM-OCR text for non-CNR pages only.
    Returns path to output txt, or None if no non-CNR pages.
    """
    doc_id = glm_info["doc_id"]
    text_dir = Path(glm_info["text_dir"])
    non_cnr_pages = sorted(
        num for num, info in page_results.items() if not info["is_cnr"]
    )

    if not non_cnr_pages:
        return None

    out_path = output_dir / f"{doc_id}_noncnr_output.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"[Non-CNR pages: {non_cnr_pages}]\n\n")

        for page_num in non_cnr_pages:
            # Find the GLM text file for this page
            # GLM saves as page-01.txt or page-1.txt
            txt_file = None
            for candidate in [
                text_dir / f"page-{page_num:02d}.txt",
                text_dir / f"page-{page_num}.txt",
            ]:
                if candidate.exists():
                    txt_file = candidate
                    break

            if txt_file:
                f.write(f"\n===== page-{page_num} =====\n")
                content = txt_file.read_text(encoding="utf-8", errors="replace")
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
            else:
                f.write(f"\n===== page-{page_num} =====\n")
                f.write(f"[GLM text not found for page {page_num}]\n")

    log(f"Non-CNR text extracted → {out_path}", "OK")
    return str(out_path)


# ── Step 3c: Combine CNR + non-CNR for mixed docs ────────────────────────────
def combine_outputs(doc_id: str, cnr_output: str, noncnr_output: str,
                    output_dir: Path) -> str:
    """Combine CNR extraction + non-CNR GLM text into final doc output."""
    out_path = output_dir / f"{doc_id}_output.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        if cnr_output and Path(cnr_output).exists():
            f.write("── CNR FORM FIELDS (Qwen extraction) ──\n")
            content = Path(cnr_output).read_text(encoding="utf-8", errors="replace")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

        if noncnr_output and Path(noncnr_output).exists():
            f.write("\n── CLINICAL / LAB DOCUMENTS (GLM-OCR text) ──\n")
            content = Path(noncnr_output).read_text(encoding="utf-8", errors="replace")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

    # Clean up intermediate files
    for p in [cnr_output, noncnr_output]:
        if p and Path(p).exists():
            Path(p).unlink()

    log(f"Combined output → {out_path}", "OK")
    return str(out_path)


# ── Step 3: Full pipeline per document ───────────────────────────────────────
def process_document(pdf_path: Path, glm_info: dict,
                     page_results: dict, output_dir: Path) -> tuple:
    """
    Route each page to correct pipeline and combine outputs.
    Returns (output_path, doc_type)
    """
    doc_id     = glm_info["doc_id"]
    cnr_pages  = [p for p, r in page_results.items() if r["is_cnr"]]
    non_pages  = [p for p, r in page_results.items() if not r["is_cnr"]]

    has_cnr    = len(cnr_pages) > 0
    has_noncnr = len(non_pages) > 0

    log(f"CNR pages: {cnr_pages} | Non-CNR pages: {non_pages}")

    if has_cnr and has_noncnr:
        doc_type = "mixed"
        log(f"STEP 3: Mixed document — Qwen on CNR pages + GLM text on non-CNR pages",
            "STEP")
    elif has_cnr:
        doc_type = "cnr_form"
        log(f"STEP 3: Pure CNR form — Qwen on all pages", "STEP")
    else:
        doc_type = "non_cnr"
        log(f"STEP 3: Non-CNR document — GLM-OCR text only", "STEP")

    cnr_output    = None
    noncnr_output = None

    # CNR pages → Qwen
    if has_cnr:
        cnr_output = run_qwen_on_cnr_pages(glm_info, page_results, output_dir)

    # Non-CNR pages → GLM text
    if has_noncnr:
        noncnr_output = extract_non_cnr_text(glm_info, page_results, output_dir)
        if not has_cnr and not cnr_output:
            # Pure non-CNR: just rename the noncnr output to final output
            final_path = output_dir / f"{doc_id}_output.txt"
            if noncnr_output and Path(noncnr_output).exists():
                Path(noncnr_output).rename(final_path)
                log(f"Non-CNR output → {final_path}", "OK")
                return str(final_path), doc_type

    # Combine both
    out_path = combine_outputs(doc_id, cnr_output, noncnr_output, output_dir)
    return out_path, doc_type


# ── Step 4: Merge all docs → patient file ────────────────────────────────────
def merge_patient_outputs(doc_outputs: list, patient_id: str,
                          output_dir: Path) -> str:
    merged_path = output_dir / f"{patient_id}_patient_raw.txt"

    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"PATIENT: {patient_id}\n")
        f.write(f"Documents: {len(doc_outputs)}\n")
        f.write(f"{'='*60}\n\n")

        for doc_info in doc_outputs:
            doc_id   = doc_info.get("doc_id", "UNKNOWN")
            doc_type = doc_info.get("type", "unknown")
            doc_path = doc_info.get("output_path")

            f.write(f"\n{'─'*50}\n")
            f.write(f"DOCUMENT: {doc_id} [{doc_type.upper()}]\n")
            f.write(f"{'─'*50}\n")

            if doc_path and Path(doc_path).exists():
                content = Path(doc_path).read_text(encoding="utf-8",
                                                    errors="replace")
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
            else:
                f.write(f"[No output for {doc_id}]\n")

    log(f"Patient merged → {merged_path}", "OK")
    return str(merged_path)


# ── Patient Registry ───────────────────────────────────────────────────────────
def get_or_assign_patient_num(patient_id_str: str) -> str:
    """
    Looks up patient_id_str in the permanent registry.
    First run  → assigns next sequential ID (001, 002...) and saves to registry.
    Re-run     → returns the same ID as before. No duplicates ever.
    Registry file: outputs/patient_registry.json
    """
    import json as _json
    registry_path = Path(WORK_DIR) / "outputs" / "patient_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = _json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}
    if patient_id_str in registry:
        return registry[patient_id_str]
    existing_nums = [int(v) for v in registry.values() if v.isdigit()]
    next_num = max(existing_nums, default=0) + 1
    patient_num = f"{next_num:03d}"
    registry[patient_id_str] = patient_num
    registry_path.write_text(
        _json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return patient_num


# ── Step 5: Anonymization ────────────────────────────────────────────────────
def run_anonymization(merged_path: str, patient_id: str,
                      output_dir: Path) -> str:
    """
    Anonymize the merged patient file.
    Produces two files:
      1. patient_raw.txt         - original merged (already exists)
      2. patient_anonymized.txt  - full text with PII replaced by [ANONYMIZED]
    Also copies anonymized file to final_anonymized/ folder.
    """
    try:
        import yaml
        with open(ANON_CONFIG, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log(f"Could not load anonymization config: {e}", "WARN")
        config = {}

    try:
        sys.path.insert(0, f"{WORK_DIR}/Anonymization_systemB")
        from anonymize_patients import extract_patient_name, anonymize_text
    except ImportError as e:
        log(f"Anonymization module not found: {e}", "WARN")
        return None

    merged_path = Path(merged_path)
    text = merged_path.read_text(encoding="utf-8", errors="replace")

    # Extract patient name
    patient_info = extract_patient_name(text)
    if patient_info.get("lastname"):
        log(f"Patient name: {patient_info['lastname']} {patient_info['firstname']}")
    else:
        log("Could not extract patient name - anonymizing without name", "WARN")

    # Get stable patient number from registry
    final_dir = Path(FINAL_OUTPUT)
    final_dir.mkdir(parents=True, exist_ok=True)
    patient_num = get_or_assign_patient_num(patient_id)

    # Anonymize
    anon_text, stats, replacements = anonymize_text(
        text, patient_info, patient_num, config
    )

    # Save anonymized file next to raw file
    anon_path = output_dir / f"{patient_id}_patient_anonymized.txt"
    anon_path.write_text(anon_text, encoding="utf-8")

    # Also copy to central final_anonymized folder
    import shutil
    final_copy = final_dir / f"patient_{patient_num}_anonymized.txt"
    shutil.copy(str(anon_path), str(final_copy))

    # Save replacement log
    import csv
    log_path = output_dir / f"{patient_id}_replacements.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category","original","replacement","start","end"]
        )
        writer.writeheader()
        writer.writerows(replacements)

    total = len(replacements)
    log(f"Anonymized → {anon_path.name} ({total} replacements)", "OK")
    log(f"Also saved → {final_copy}", "OK")
    for k, v in sorted(stats.items()):
        log(f"  {k}: {v}")

    return str(anon_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def process_patient(folder: str, debug: bool = False) -> str:
    folder     = Path(folder)
    patient_id = get_patient_id(folder)

    print(f"\n{'╔'+'═'*56+'╗'}")
    print(f"║  PATIENT : {patient_id:<46}║")
    print(f"║  Folder  : {str(folder.name):<46}║")
    print(f"╚{'═'*56}╝")

    output_dir = Path(OUTPUT_BASE) / patient_id.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".pdf")
    if not pdfs:
        log(f"No PDFs in {folder}", "WARN")
        return None

    log(f"Found {len(pdfs)} PDF(s)")
    doc_outputs = []

    for i, pdf_path in enumerate(pdfs, 1):
        doc_id = pdf_path.stem
        print(f"\n  ┌{'─'*50}")
        print(f"  │  [{i}/{len(pdfs)}] {doc_id}")
        print(f"  └{'─'*50}")

        # Step 1: GLM-OCR
        log("STEP 1: GLM-OCR", "STEP")
        glm_info = run_glm_ocr(pdf_path)
        if not glm_info.get("success"):
            log(f"Skipping {doc_id} (GLM-OCR failed)", "ERROR")
            doc_outputs.append({"doc_id": doc_id, "type": "failed",
                                 "output_path": None})
            continue

        # Step 2: Classify pages
        log("STEP 2: Classify pages", "STEP")
        page_results = classify_pages(glm_info.get("pages_dir", ""),
                                      debug=debug)
        if not page_results:
            log(f"No pages found for {doc_id}", "WARN")
            continue

        # Step 3: Process document
        out_path, doc_type = process_document(
            pdf_path, glm_info, page_results, output_dir
        )

        doc_outputs.append({"doc_id": doc_id, "type": doc_type,
                             "output_path": out_path})

    # Step 4: Merge
    log("STEP 4: Merging all documents", "STEP")
    merged = merge_patient_outputs(doc_outputs, patient_id, output_dir)

    # Step 5: Anonymize merged file
    log("STEP 5: Anonymizing patient file", "STEP")
    anonymized_path = run_anonymization(merged, patient_id, output_dir)

    cnr_n    = sum(1 for d in doc_outputs if d["type"] == "cnr_form")
    mixed_n  = sum(1 for d in doc_outputs if d["type"] == "mixed")
    noncnr_n = sum(1 for d in doc_outputs if d["type"] == "non_cnr")
    failed_n = sum(1 for d in doc_outputs if d["type"] == "failed")

    print(f"\n{'╔'+'═'*56+'╗'}")
    print(f"║  DONE: {patient_id:<50}║"[:61] + "║")
    print(f"║  CNR docs     : {cnr_n:<40}║")
    print(f"║  Mixed docs   : {mixed_n:<40}║")
    print(f"║  Non-CNR docs : {noncnr_n:<40}║")
    print(f"║  Failed       : {failed_n:<40}║")
    print(f"╚{'═'*56}╝\n")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = process_patient(args.folder, debug=args.debug)
    if result:
        print(f"✅ Patient file: {result}")
        sys.exit(0)
    else:
        print("❌ Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
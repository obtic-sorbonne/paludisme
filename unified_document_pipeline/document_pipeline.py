from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import subprocess
from collections import defaultdict


ROOT_DIR = Path("/home/lfarooq/digitize_medical_records")
BENCHMARK_DIR = ROOT_DIR / "benchmark_outputs"


def clean_text(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_cmd(cmd: list[str], log_lines: list[str], cwd: Path | None = None):
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {cmd_str}", flush=True)
    log_lines.append(f"$ {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
    )

    if result.returncode != 0:
        log_lines.append(f"[ERROR] Command failed ({result.returncode}): {cmd_str}")
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd_str}")


def run_cmd_soft(cmd: list[str], log_lines: list[str], cwd: Path | None = None):
    cmd_str = " ".join(cmd)
    print(f"\n[RUN-SOFT] {cmd_str}", flush=True)
    log_lines.append(f"$ {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
    )

    ok = result.returncode == 0
    if not ok:
        warning = f"[WARNING] Soft-fail command ({result.returncode}): {cmd_str}"
        print(warning, flush=True)
        log_lines.append(warning)

    return ok, result.returncode


def run_anonymization_on_merged_output(
    merged_txt_path: Path,
    doc_out_dir: Path,
    log_lines: list[str],
):
    anonymized_txt_path = doc_out_dir / "merged_final_output_anonymized.txt"

    cmd = [
        "python",
        "-m",
        "anonymization.test_single_file",
        str(merged_txt_path),
        "-o",
        str(anonymized_txt_path),
    ]

    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {cmd_str}", flush=True)
    log_lines.append(f"$ {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        text=True,
    )

    if result.returncode != 0:
        warning = f"[WARNING] Anonymization failed ({result.returncode}) for {merged_txt_path.name}"
        print(warning, flush=True)
        log_lines.append(warning)
        return None, None

    repl_path = anonymized_txt_path.with_suffix(".replacements.txt")
    return anonymized_txt_path, repl_path

def extract_page_num_from_route_name(path: Path) -> int | None:
    m = re.search(r"_(\d+)_(\d+)_res_route\.json$", path.name)
    if not m:
        return None
    return int(m.group(1)) + 1


def load_route_files_for_doc(doc_stem: str):
    route_dir = BENCHMARK_DIR / "page_routing"
    files = sorted(route_dir.glob(f"{doc_stem}_*_res_route.json"))
    routes = []

    for rf in files:
        data = load_json(rf)
        page_num = extract_page_num_from_route_name(rf)
        if page_num is None:
            continue
        routes.append(
            {
                "page_num": page_num,
                "route_file": str(rf),
                "data": data,
            }
        )

    routes.sort(key=lambda x: x["page_num"])
    return routes


def parse_pages_from_txt(txt_path: Path) -> dict[int, list[str]]:
    if not txt_path.exists():
        return {}

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    pages: dict[int, list[str]] = {}
    current_page = None
    current_lines: list[str] = []

    page_pat = re.compile(r"^===== PAGE (\d+) =====$")

    for raw in lines:
        line = raw.rstrip("\n")
        m = page_pat.match(line.strip())
        if m:
            if current_page is not None:
                pages[current_page] = current_lines[:]
            current_page = int(m.group(1))
            current_lines = []
        else:
            if current_page is not None:
                current_lines.append(line)

    if current_page is not None:
        pages[current_page] = current_lines[:]

    return pages


def cleanup_old_lab_outputs(doc_stem: str, page_num: int):
    page_index = page_num - 1

    candidates = [
        BENCHMARK_DIR / "hybrid_tables_final" / f"{doc_stem}_{page_index}_res_hybrid_final.txt",
        BENCHMARK_DIR / "hybrid_tables" / f"{doc_stem}_{page_index}_res_hybrid.txt",
        BENCHMARK_DIR / "lab_fallback_text" / f"{doc_stem}_{page_index}_{page_index}_ocr_fallback.txt",
        BENCHMARK_DIR / "lab_fallback_meta" / f"{doc_stem}_{page_index}_{page_index}_fallback.json",
    ]

    for p in candidates:
        if p.exists():
            p.unlink()


def get_lab_output_paths(doc_stem: str, page_num: int):
    page_index = page_num - 1

    final_hybrid = BENCHMARK_DIR / "hybrid_tables_final" / f"{doc_stem}_{page_index}_res_hybrid_final.txt"
    hybrid = BENCHMARK_DIR / "hybrid_tables" / f"{doc_stem}_{page_index}_res_hybrid.txt"
    fallback_txt = BENCHMARK_DIR / "lab_fallback_text" / f"{doc_stem}_{page_index}_{page_index}_ocr_fallback.txt"

    return final_hybrid, hybrid, fallback_txt


def extract_lab_output_for_page(doc_stem: str, page_num: int, branch_failures: dict[int, str]) -> str:
    final_hybrid, hybrid, fallback_txt = get_lab_output_paths(doc_stem, page_num)

    if final_hybrid.exists():
        return final_hybrid.read_text(encoding="utf-8").strip()

    if hybrid.exists():
        return hybrid.read_text(encoding="utf-8").strip()

    if fallback_txt.exists():
        return fallback_txt.read_text(encoding="utf-8").strip()

    if page_num in branch_failures:
        return f"LAB PIPELINE FAILED FOR PAGE {page_num}: {branch_failures[page_num]}"

    return f"LAB OUTPUT NOT FOUND FOR PAGE {page_num}"


def load_form_outputs(doc_stem: str) -> tuple[str | None, dict | None]:
    txt_path = BENCHMARK_DIR / "full_document_all_pages_parser" / f"{doc_stem}_full_final_output.txt"
    json_path = BENCHMARK_DIR / "full_document_all_pages_parser" / f"{doc_stem}_all_pages_all_specs.json"

    txt_content = None
    json_content = None

    if txt_path.exists():
        txt_content = txt_path.read_text(encoding="utf-8").strip()

    if json_path.exists():
        json_content = load_json(json_path)

    return txt_content, json_content


def build_merged_output(
    pdf_path: Path,
    routes: list[dict],
    narrative_pages: dict[int, list[str]],
    doc_stem: str,
    branch_failures: dict[int, str],
    form_output_txt: str | None,
    form_output_json: dict | None,
):
    merged_txt_lines = []
    merged_json = {
        "pdf_path": str(pdf_path),
        "doc_stem": doc_stem,
        "pages": [],
        "branch_failures": branch_failures,
    }

    merged_txt_lines.append(f"PDF: {pdf_path}")
    merged_txt_lines.append(f"DOC_STEM: {doc_stem}")
    merged_txt_lines.append("")

    form_block_written = False

    for item in routes:
        page_num = item["page_num"]
        route = item["data"]["routing"]
        primary_type = route.get("primary_page_type", "unknown")
        next_step = route.get("next_step", "keep_text_only")

        page_entry = {
            "page_num": page_num,
            "primary_page_type": primary_type,
            "next_step": next_step,
            "content_source": None,
            "content": None,
        }

        merged_txt_lines.append("=" * 80)
        merged_txt_lines.append(f"PAGE {page_num}")
        merged_txt_lines.append(f"TYPE: {primary_type}")
        merged_txt_lines.append(f"NEXT STEP: {next_step}")
        merged_txt_lines.append("-" * 80)

        if next_step == "lab_table_extraction":
            final_hybrid, hybrid, fallback_txt = get_lab_output_paths(doc_stem, page_num)
            content = extract_lab_output_for_page(doc_stem, page_num, branch_failures)
            merged_txt_lines.append(content)

            if final_hybrid.exists():
                page_entry["content_source"] = "lab_tables_pipeline_final"
            elif hybrid.exists():
                page_entry["content_source"] = "lab_tables_pipeline_hybrid"
            elif fallback_txt.exists():
                page_entry["content_source"] = "lab_tables_pipeline_fallback_ocr"
            else:
                page_entry["content_source"] = "lab_tables_pipeline_missing"

            page_entry["content"] = content

        elif next_step == "report_text_extraction":
            lines = narrative_pages.get(page_num, [])
            content = "\n".join(lines).strip() if lines else f"NARRATIVE OCR OUTPUT NOT FOUND FOR PAGE {page_num}"
            merged_txt_lines.append(content)
            page_entry["content_source"] = "narrative_clinical_pipeline"
            page_entry["content"] = content

        elif next_step == "form_field_extraction":
            if not form_block_written:
                if form_output_txt:
                    content = form_output_txt
                    merged_txt_lines.append(content)
                    page_entry["content_source"] = "checkbox_full_doc_pipeline"
                    page_entry["content"] = content
                    if form_output_json is not None:
                        page_entry["structured_json"] = form_output_json
                else:
                    content = f"FORM STRUCTURED OUTPUT NOT FOUND FOR DOC {doc_stem}"
                    merged_txt_lines.append(content)
                    page_entry["content_source"] = "checkbox_full_doc_pipeline_missing_output"
                    page_entry["content"] = content
                form_block_written = True
            else:
                content = "[FORM CONTENT ALREADY INCLUDED ABOVE FROM CHECKBOX FULL-DOCUMENT PIPELINE]"
                merged_txt_lines.append(content)
                page_entry["content_source"] = "checkbox_full_doc_pipeline_reference"
                page_entry["content"] = content

        else:
            lines = narrative_pages.get(page_num, [])
            content = "\n".join(lines).strip() if lines else f"RAW OCR TEXT NOT FOUND FOR PAGE {page_num}"
            merged_txt_lines.append(content)
            page_entry["content_source"] = "keep_text_only_from_narrative_ocr"
            page_entry["content"] = content

        merged_txt_lines.append("")
        merged_json["pages"].append(page_entry)

    return "\n".join(merged_txt_lines), merged_json


def main():
    parser = argparse.ArgumentParser(
        description="Unified document pipeline: classify pages, dispatch to correct branch, merge outputs."
    )
    parser.add_argument("pdf_path", help="Full path to the PDF")
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_DIR / "final_document_pipeline"),
        help="Base directory for unified outputs",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    doc_stem = pdf_path.stem
    base_out_dir = Path(args.output_dir)
    doc_out_dir = base_out_dir / doc_stem
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    log_lines.append("UNIFIED DOCUMENT PIPELINE")
    log_lines.append("=" * 80)
    log_lines.append(f"PDF: {pdf_path}")
    log_lines.append(f"DOC_STEM: {doc_stem}")
    log_lines.append("")

    branch_failures: dict[int, str] = {}

    run_cmd(
        [
            str(ROOT_DIR / "page_classification_pipeline" / "run_page_classification_pipeline.sh"),
            str(pdf_path),
        ],
        log_lines,
        cwd=ROOT_DIR,
    )

    routes = load_route_files_for_doc(doc_stem)
    if not routes:
        raise RuntimeError(f"No route JSON files found for {doc_stem}")

    grouped_pages = defaultdict(list)
    for item in routes:
        grouped_pages[item["data"]["routing"]["next_step"]].append(item["page_num"])

    log_lines.append("")
    log_lines.append("PAGE GROUPS")
    for step, pages in grouped_pages.items():
        log_lines.append(f"{step}: {pages}")

    narrative_pages = {}
    need_narrative = (
        grouped_pages.get("report_text_extraction")
        or grouped_pages.get("keep_text_only")
    )

    if need_narrative:
        run_cmd(
            [
                str(ROOT_DIR / "narrative_clinical_pipeline" / "run_narrative_layout_pipeline.sh"),
                str(pdf_path),
            ],
            log_lines,
            cwd=ROOT_DIR,
        )

        narrative_txt = BENCHMARK_DIR / "paddle" / f"{doc_stem}.txt"
        narrative_pages = parse_pages_from_txt(narrative_txt)

    form_output_txt = None
    form_output_json = None

    if grouped_pages.get("form_field_extraction"):
        run_cmd(
            [
                str(ROOT_DIR / "checkbox_full_doc_pipeline" / "run_checkbox_form_pipeline.sh"),
                str(pdf_path),
            ],
            log_lines,
            cwd=ROOT_DIR,
        )
        form_output_txt, form_output_json = load_form_outputs(doc_stem)

    for page_num in grouped_pages.get("lab_table_extraction", []):
        cleanup_old_lab_outputs(doc_stem, page_num)

        page_index = page_num - 1
        ok, code = run_cmd_soft(
            [
                str(ROOT_DIR / "lab_tables_pipeline" / "run_lab_table_pipeline.sh"),
                str(pdf_path),
                str(page_index),
            ],
            log_lines,
            cwd=ROOT_DIR,
        )
        if not ok:
            branch_failures[page_num] = f"run_lab_table_pipeline.sh failed with code {code}"

    merged_txt, merged_json = build_merged_output(
        pdf_path=pdf_path,
        routes=routes,
        narrative_pages=narrative_pages,
        doc_stem=doc_stem,
        branch_failures=branch_failures,
        form_output_txt=form_output_txt,
        form_output_json=form_output_json,
    )

    merged_txt_path = doc_out_dir / "merged_final_output.txt"
    merged_json_path = doc_out_dir / "merged_final_output.json"
    log_path = doc_out_dir / "pipeline.log"

    merged_txt_path.write_text(merged_txt, encoding="utf-8")
    save_json(merged_json_path, merged_json)

    print(f"Saved merged TXT:  {merged_txt_path}")
    print(f"Saved merged JSON: {merged_json_path}")

    anonymized_txt_path, repl_path = run_anonymization_on_merged_output(
        merged_txt_path=merged_txt_path,
        doc_out_dir=doc_out_dir,
        log_lines=log_lines,
    )

    if anonymized_txt_path:
        print(f"Saved anonymized TXT: {anonymized_txt_path}")
    if repl_path and repl_path.exists():
        print(f"Saved replacements:   {repl_path}")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Saved log:            {log_path}")

if __name__ == "__main__":
    main()
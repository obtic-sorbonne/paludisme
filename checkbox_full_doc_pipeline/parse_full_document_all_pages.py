from pathlib import Path
import argparse
import json
import re

import cv2

from parse_page4_controle_parasito_visual import (
    load_json as load_controle_visual_json,
    extract_page_words as extract_controle_visual_words,
    parse_controle_block as parse_controle_block_visual,
)

from parse_radio_visual_from_ocr_json import (
    load_json as load_visual_json,
    extract_page_words as extract_visual_words,
    parse_radio_field,
)
from parse_page3_from_ocr_text import (
    page3_specs,
    parse_protection_personnelle_block,
    parse_chimioprophylaxie_block,
    parse_bandelettes_block,
)
from cnr_common import (
    load_ocr_txt,
    clean_text,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    norm,
    compact_norm,
)

from parse_page1_from_ocr_text import page1_specs
from parse_page2_from_ocr_text import page2_specs

from parse_page4_from_ocr_text import (
    page4_specs,
    parse_controle_parasitologique,
)


# --------------------------------------------------
# OCR page extraction
# --------------------------------------------------

def extract_all_pages(lines: list[str]) -> list[dict]:
    pages = []
    current_page = None
    current_lines = []

    page_pat = re.compile(r"^===== PAGE (\d+) =====$")

    for raw in lines:
        line = clean_text(raw)
        m = page_pat.match(line)

        if m:
            if current_page is not None:
                pages.append({
                    "page_num": current_page,
                    "lines": [clean_text(x) for x in current_lines if clean_text(x)],
                })
            current_page = int(m.group(1))
            current_lines = []
        else:
            current_lines.append(raw)

    if current_page is not None:
        pages.append({
            "page_num": current_page,
            "lines": [clean_text(x) for x in current_lines if clean_text(x)],
        })

    return pages


def preprocess_page_lines(page_num: int, lines: list[str]) -> list[str]:
    """Apply all known OCR corrections then run postprocess_lines."""
    replacements = [
        ("Afriain", "Africain"),
        ("Ethnicite", "Ethnicité"),
        ("Duree du sejour", "Durée du séjour"),
        ("endemie", "endémie"),
        ("adresse parun", "adressé par un"),
        ("recherchebiologiquede", "recherche biologique de"),
        ("Paludismea-t-elle", "Paludisme a-t-elle"),
        ("Acette", "A cette"),
        ("NoN", "NON"),
        ("Si OUl,resultat", "Si OUI, résultat"),
        ("Si OUl, resultat", "Si OUI, résultat"),
        ("Si oul,a quelle date", "Si OUI, à quelle date"),
        ("Paysderésidence", "Pays de résidence"),
        ("Paysd'endémie", "Pays d'endémie"),
        ("Datedepart", "Date départ"),
        ("Datede naissance", "Date de naissance"),
        ("Fréquence des sejours", "Fréquence des séjours"),
        ("Residence durant", "Résidence durant"),
        ("acces", "accès"),
        ("ONSP", "O NSP"),
        ("Povale", "P ovale"),
        ("Pvivax", "P vivax"),
        ("XHospitalisation", "X Hospitalisation"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
        ("O UI", "OUI"),
        ("O U I", "OUI"),
        ("demière", "dernière"),
        ("Symptomes", "Symptômes"),
        ("OAccès", "O Accès"),
        ("OMilitaires", "O Militaires"),
    ]

    out = []
    for line in lines:
        line = clean_text(line)
        for old, new in replacements:
            line = line.replace(old, new)
        out.append(line)

    return postprocess_lines(out)


# --------------------------------------------------
# Visual fallback helpers
# --------------------------------------------------

def get_page_json_and_image_paths(ocr_txt_path: Path, page_num: int):
    """
    Map OCR TXT document + page number to:
      - per-page Paddle JSON
      - rendered page PNG
    Assumes naming:
      DOC_00116_0_0_res.json for PDF page 1
      DOC_00116-1.png for PDF page 1
    """
    stem = ocr_txt_path.stem

    json_path = Path(
        f"/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/{stem}_{page_num - 1}_{page_num - 1}_res.json"
    )
    image_path = Path(
        f"/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages/{stem}-{page_num}.png"
    )

    return json_path, image_path


VISUAL_FIELD_CONFIG = {
    "Consultation avant": {
        "field_key": "consultation_avant",
        "target_page": 2,
    },
    "Résidence en zone d'endémie": {
        "field_key": "residence_zone_endemie",
        "target_page": 1,
    },
    "Etat clinique au moment du diagnostic": {
        "field_key": "etat_clinique",
        "target_page": 2,
    },
    "Chimioprophylaxie utilisée": {
        "field_key": "chimioprophylaxie_oui_non_nsp",
        "target_page": 3,
    },
    "Evolution clinique": {
        "field_key": "evolution_clinique",
        "target_page": 4,
    },
    "Sexe": {
        "field_key": "sexe",
        "target_page": 1,
    },
}


def run_visual_fallback_for_field(ocr_txt_path: Path, field_name: str):
    """
    Run targeted visual fallback for one field.
    Returns selected option string or None.
    """
    cfg = VISUAL_FIELD_CONFIG.get(field_name)
    if not cfg:
        return None

    page_num = cfg["target_page"]
    field_key = cfg["field_key"]

    json_path, image_path = get_page_json_and_image_paths(ocr_txt_path, page_num)

    if not json_path.exists() or not image_path.exists():
        return None

    try:
        data = load_visual_json(json_path)
        data = data.get("res", data)
        words = extract_visual_words(data, page_num=1)

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None

        debug_dir = Path(
            "/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/visual_debug"
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        page_stem = f"{ocr_txt_path.stem}_page{page_num}"
        result = parse_radio_field(words, img_bgr, field_key, debug_dir, page_stem)

        return result.get("selected_option")
    except Exception:
        return None

def run_controle_parasito_visual_fallback(ocr_txt_path: Path):
    """
    Run visual fallback for the contrôle parasitologique block.
    This block is on OCR page 5 for DOC_00116-like files.
    """
    page_num = 5
    json_path, image_path = get_page_json_and_image_paths(ocr_txt_path, page_num)

    if not json_path.exists() or not image_path.exists():
        return None

    try:
        data = load_controle_visual_json(json_path)
        data = data.get("res", data)
        words = extract_controle_visual_words(data, page_num=1)
        result = parse_controle_block_visual(words)
        return result
    except Exception:
        return None
    
def apply_visual_fallbacks(ocr_txt_path: Path, merged_fields: dict) -> dict:
    """
    Override missing/weak TXT results with visual parser results
    for selected stable fields.
    """
    out = {}

    for field, fr in merged_fields.items():
        item = dict(fr)
        selected = item.get("selected_options", [])
        found = item.get("found", False)

        should_try_visual = (
            field in VISUAL_FIELD_CONFIG
            and (
                not selected
                or not found
            )
        )

        if should_try_visual:
            visual_selected = run_visual_fallback_for_field(ocr_txt_path, field)
            if visual_selected:
                item["selected_options"] = [visual_selected]
                item["visual_override"] = True
                item["visual_selected_option"] = visual_selected
                item["found"] = True
                item["source"] = f"{item.get('source', 'unknown')} + visual_fallback"

        out[field] = item

    return out


def apply_nature_du_sejour_rule(merged_fields: dict, results: list[dict]) -> dict:
    """
    If 'Nature du séjour' has no selected option, but the document contains
    a filled 'Si autres, préciser' text field, infer 'Autres'.

    This is a form-structure rule, not a document-specific hardcoded answer.
    """
    out = {k: dict(v) for k, v in merged_fields.items()}

    field_name = "Nature du séjour"
    if field_name not in out:
        return out

    current = out[field_name]
    already_selected = current.get("selected_options", [])
    if already_selected:
        return out

    filled_autres_text = None
    found_page = None

    for page_result in results:
        page_num = page_result["page_num"]
        full_lines = page_result.get("processed_lines", [])

        for i, line in enumerate(full_lines):
            c = compact_norm(line)

            # robust detection of "Si autres, préciser"
            if "siautrespreciser" in c:
                # Case 1: value is on same line after colon
                if ":" in line:
                    right = clean_text(line.split(":", 1)[1])
                    if right:
                        filled_autres_text = right
                        found_page = page_num
                        break

                # Case 2: value may be on next line
                if i + 1 < len(full_lines):
                    nxt = clean_text(full_lines[i + 1])
                    if nxt and "similitaire" not in compact_norm(nxt):
                        filled_autres_text = nxt
                        found_page = page_num
                        break

        if filled_autres_text:
            break

    if filled_autres_text:
        current["selected_options"] = ["Autres"]
        current["found"] = True
        current["nature_du_sejour_inferred_from_autres_preciser"] = filled_autres_text
        current["source"] = f"{current.get('source', 'unknown')} + autres_preciser_rule"
        current["page_num"] = found_page or current.get("page_num")

    out[field_name] = current
    return out

# --------------------------------------------------
# Structured parsing
# --------------------------------------------------

def run_specs(lines: list[str], specs: list[dict], source_name: str):
    sections = []
    field_results = []

    for spec in specs:
        sec = slice_section(
            lines=lines,
            start_anchors=spec["start_anchors"],
            end_anchors=spec.get("end_anchors"),
            max_lines=spec.get("max_lines", 20),
        )

        if sec is None:
            sections.append({
                "field": spec["field"],
                "found": False,
                "source": source_name,
            })
            field_results.append({
                "field": spec["field"],
                "found": False,
                "source": source_name,
                "selected_options": [],
                "options": [],
            })
            continue

        sections.append({
            "field": spec["field"],
            "found": True,
            "source": source_name,
            "anchor_line": sec["anchor_line"],
            "start_idx": sec["start_idx"],
            "end_idx": sec["end_idx"],
            "lines": sec["lines"],
        })

        option_results = []
        for canonical, variants in spec["options"]:
            option_results.append(
                parse_option_from_lines(sec["lines"], canonical, variants)
            )

        option_results = postprocess_single_choice(option_results)
        option_results = apply_elimination_heuristic(
            option_results,
            spec.get("single_choice", False),
        )
        selected_options = [x["option"] for x in option_results if x.get("selected")]

        field_results.append({
            "field": spec["field"],
            "found": True,
            "source": source_name,
            "selected_options": selected_options,
            "options": option_results,
        })

    return sections, field_results


def safe_custom_block(func, lines: list[str], block_name: str):
    try:
        result = func(lines)
        return {
            "block": block_name,
            "success": True,
            "result": result,
        }
    except Exception as e:
        return {
            "block": block_name,
            "success": False,
            "error": str(e),
        }


def run_all_specs_on_page(lines: list[str]):
    all_sections = []
    all_field_results = []

    spec_groups = [
        ("page1_specs", page1_specs()),
        ("page2_specs", page2_specs()),
        ("page3_specs", page3_specs()),
        ("page4_specs", page4_specs()),
    ]

    for source_name, specs in spec_groups:
        sections, field_results = run_specs(lines, specs, source_name)
        all_sections.extend(sections)
        all_field_results.extend(field_results)

    custom_blocks = [
    safe_custom_block(
        parse_protection_personnelle_block,
        lines,
        "parse_protection_personnelle_block",
    ),
    safe_custom_block(
        parse_chimioprophylaxie_block,
        lines,
        "parse_chimioprophylaxie_block",
    ),
    safe_custom_block(
        parse_bandelettes_block,
        lines,
        "parse_bandelettes_block",
    ),
    safe_custom_block(
        parse_controle_parasitologique,
        lines,
        "parse_controle_parasitologique",
    ),
]

    return {
        "sections": all_sections,
        "field_results": all_field_results,
        "custom_blocks": custom_blocks,
    }


# --------------------------------------------------
# Field merging across pages
# --------------------------------------------------

def merge_across_pages(results: list[dict]) -> dict:
    """
    Merge field_results across all pages, preferring the page that actually
    found options and selected something.
    """
    merged: dict[str, dict] = {}

    for page_result in results:
        page_num = page_result["page_num"]
        for fr in page_result["all_page_parsers_output"]["field_results"]:
            field = fr["field"]
            if field not in merged:
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num
                continue

            existing = merged[field]

            if fr.get("selected_options") and not existing.get("selected_options"):
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num
            elif fr.get("found") and not existing.get("found"):
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num

    return merged


# --------------------------------------------------
# Main document runner
# --------------------------------------------------

def custom_block_has_meaningful_result(res: dict) -> bool:
    field = res.get("field")

    if field == "Bandelettes":
        return bool(res.get("found")) and (
            bool(res.get("status")) or
            bool(res.get("selected_results")) or
            bool(res.get("all_results_found"))
        )

    if field == "Protection Personnelle Anti-Moustiques":
        if res.get("selected"):
            return True
        for d in res.get("details", []):
            if d.get("selection") or d.get("options_found"):
                return True
        return False

    if field == "Chimioprophylaxie utilisée":
        return bool(res.get("found")) and (
            bool(res.get("selected")) or
            bool(res.get("table_results"))
        )

    if field == "Contrôle parasitologique P falciparum":
        if res.get("control_overall"):
            return True
        for row in res.get("rows", []):
            if (
                row.get("fait") or
                row.get("temperature") or
                row.get("parasitologie") or
                row.get("densite_parasitaire")
            ):
                return True
        return False

    return bool(res.get("found"))


def collect_meaningful_custom_blocks(ocr_txt_path: Path, results: list[dict]) -> list[dict]:
    """
    Collect meaningful custom block results from all pages.
    If contrôle parasitologique is missing/empty from text parsing,
    try the visual fallback parser.
    """
    meaningful = []

    for page_result in results:
        page_num = page_result["page_num"]
        parsed = page_result.get("all_page_parsers_output", {})
        custom_blocks = parsed.get("custom_blocks", [])

        for block in custom_blocks:
            if not block.get("success", False):
                continue
            res = block.get("result", {})
            if custom_block_has_meaningful_result(res):
                meaningful.append({
                    "page_num": page_num,
                    "result": res,
                })

    # if no meaningful contrôle parasitologique block exists, try visual fallback
    has_controle = any(
        x["result"].get("field") == "Contrôle parasitologique P falciparum"
        for x in meaningful
    )

    if not has_controle:
        vis_res = run_controle_parasito_visual_fallback(ocr_txt_path)
        if vis_res and custom_block_has_meaningful_result(vis_res):
            meaningful.append({
                "page_num": 5,
                "result": vis_res,
            })

    return meaningful


def build_full_document_txt(ocr_txt_path: Path, pages: list[dict], merged_fields: dict, results: list[dict]) -> str:
    """
    Build a clean final TXT containing only final extracted results.
    No raw OCR lines.
    """
    lines = []

    lines.append(f"OCR TXT PATH: {ocr_txt_path}")
    lines.append(f"Physical pages found: {len(pages)}")
    lines.append("")

    lines.append("=== FINAL STRUCTURED ANSWERS ===")
    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)

        if sel:
            value = ", ".join(sel)
        elif found:
            value = "None"
        else:
            value = "None"

        lines.append(f"{field}: {value}")

    lines.append("")
    lines.append("=== CUSTOM BLOCKS ===")

    meaningful_blocks = collect_meaningful_custom_blocks(ocr_txt_path, results)

    if not meaningful_blocks:
        lines.append("None")
        return "\n".join(lines)

    for item in meaningful_blocks:
        res = item["result"]
        field = res.get("field", "unknown_block")

        if field == "Protection Personnelle Anti-Moustiques":
            selected = res.get("selected")
            lines.append(f"{field}: {selected if selected else 'None'}")
            for d in res.get("details", []):
                item_name = d.get("item")
                selection = d.get("selection")
                if selection:
                    lines.append(f"  {item_name}: {selection}")
            continue

        if field == "Chimioprophylaxie utilisée":
            selected = res.get("selected")
            lines.append(f"{field}: {selected if selected else 'None'}")
            table_results = res.get("table_results", [])
            for row in table_results:
                med = row.get("medicine")
                sel = row.get("selection")
                lines.append(f"  {med}: {sel if sel else 'None'}")
            continue

        if field == "Bandelettes":
            status = res.get("status")
            selected_results = res.get("selected_results", [])
            value_parts = []
            if status:
                value_parts.append(status)
            if selected_results:
                value_parts.append(", ".join(selected_results))
            lines.append(f"{field}: {' | '.join(value_parts) if value_parts else 'None'}")
            continue

        if field == "Contrôle parasitologique P falciparum":
            overall = res.get("control_overall")
            lines.append(f"{field}: {overall if overall else 'None'}")
            for row in res.get("rows", []):
                row_name = row.get("row")
                fait = row.get("fait")
                temp = row.get("temperature")
                paras = ", ".join(row.get("parasitologie", [])) if row.get("parasitologie") else "None"
                dens = ", ".join(row.get("densite_parasitaire", [])) if row.get("densite_parasitaire") else "None"

                lines.append(f"  {row_name}: {fait if fait else 'None'}")
                lines.append(f"    Température: {temp if temp else 'None'}")
                lines.append(f"    Parasitologie: {paras}")
                lines.append(f"    Densité parasitaire: {dens}")
            continue

        lines.append(f"{field}: {json.dumps(res, ensure_ascii=False)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Apply all page specs to every page in OCR txt"
    )
    parser.add_argument("ocr_txt_path", help="Path to OCR txt file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_lines = load_ocr_txt(ocr_txt_path)
    pages = extract_all_pages(raw_lines)

    results = []

    for page in pages:
        processed_lines = preprocess_page_lines(page["page_num"], page["lines"])
        parsed = run_all_specs_on_page(processed_lines)

        results.append({
            "page_num": page["page_num"],
            "lines_count": len(processed_lines),
            "processed_lines": processed_lines,
            "processed_lines_preview": processed_lines[:120],
            "all_page_parsers_output": parsed,
        })

    merged_fields = merge_across_pages(results)
    merged_fields = apply_visual_fallbacks(ocr_txt_path, merged_fields)
    merged_fields = apply_nature_du_sejour_rule(merged_fields, results)

    final_result = {
        "ocr_txt_path": str(ocr_txt_path),
        "pages_detected": [p["page_num"] for p in pages],
        "pages_count": len(pages),
        "merged_field_results": merged_fields,
        "results": results,
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_all_pages_all_specs.json"
    out_json.write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = []
    summary_lines.append(f"Pages detected: {len(pages)}")
    summary_lines.append(f"Merged fields:  {len(merged_fields)}")
    summary_lines.append(f"Saved JSON: {out_json}")
    summary_lines.append("")
    summary_lines.append("=== MERGED SELECTIONS SUMMARY ===")

    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)
        page = fr.get("page_num", "?")
        status = ", ".join(sel) if sel else ("(found, no selection)" if found else "(not found)")
        summary_lines.append(f"  [{page}] {field}: {status}")

    out_txt = out_dir / f"{ocr_txt_path.stem}_all_pages_all_specs_summary.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved summary TXT: {out_txt}")
    print(f"Pages detected: {len(pages)}")
    print(f"Merged fields:  {len(merged_fields)}")
    print(f"Saved JSON: {out_json}")

    print("\n=== MERGED SELECTIONS SUMMARY ===")
    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)
        page = fr.get("page_num", "?")
        status = ", ".join(sel) if sel else ("(found, no selection)" if found else "(not found)")
        print(f"  [{page}] {field}: {status}")

    full_txt = build_full_document_txt(ocr_txt_path, pages, merged_fields, results)
    out_full_txt = out_dir / f"{ocr_txt_path.stem}_full_final_output.txt"
    out_full_txt.write_text(full_txt, encoding="utf-8")
    print(f"Saved full final TXT: {out_full_txt}")

if __name__ == "__main__":
    main()
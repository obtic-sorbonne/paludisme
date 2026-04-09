from pathlib import Path
import argparse
import re


JUNK_TOKENS = {"回", "国", "□", "网", "\\", "……", "", "V", "√"}


def norm(s: str) -> str:
    return " ".join(
        s.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ï", "i")
        .replace("î", "i")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("'", "'")
        .split()
    )


def clean_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def is_metadata_line(line: str) -> bool:
    t = norm(line)
    keys = [
        "nom patient",
        "date / heure",
        "date heure",
        "date naissance",
        "prescripteur",
        "adresse",
        "patient adresse",
        "copie a",
        "echantillon",
        "prelevement",
        "demande",
        "resultats d'une demande",
        "consultit",
        "75019 paris",
        "75010 paris 10",
        "tel:",
        "hopital robert debre",
        "lae",
        "requestresults.aspx",
        "48 bd serrurier",
        "valide par",
        "aubervilliers",
        "robert debre",
    ]
    return any(k in t for k in keys)


def is_header_line(line: str) -> bool:
    t = norm(line)
    keys = ["description", "resultat", "unite", "valeurs normales", "valeursnormales", "val."]
    return sum(1 for k in keys if k in t) >= 2


def is_section_title(line: str) -> bool:
    t = norm(line)
    keys = [
        "examens d'hematologie",
        "cytologie",
        "biochimie generale",
        "examens de sang",
        "gaz du sang",
    ]
    return any(k in t for k in keys)


def sanitize_columns(cols):
    out = []
    for c in cols:
        c = clean_text(c)
        if c in JUNK_TOKENS:
            c = ""
        out.append(c)
    return out


def is_biochemistry_table_output(core_lines):
    """Detect if the core table contains biochemistry data"""
    joined = " ".join(core_lines)
    j = norm(joined)
    
    biochem_markers = [
        "sodium", "potassium", "chlore", "bicarbonates",
        "proteines plasmatiques", "uree", "creatinine", "glycemie",
        "phosphatases alcalines", "bilirubine", "asat", "alat", "ggt",
        "prealbumine", "hemolyse", "lipemie"
    ]
    
    hits = sum(1 for m in biochem_markers if m in j)
    return hits >= 3


def fix_merged_result_unit_hematology(row):
    """Fix merged columns for hematology tables (4 columns: desc|result|unit|normal)"""
    row = sanitize_columns(row)
    if len(row) < 5:
        row = row + [""] * (5 - len(row))

    desc, result, unit, normal, val = row[:5]

    # Case 1: desc + qualitative result merged
    if desc and not result:
        m = re.match(
            r"^(.*?)(?:\s+)(non|oui|pos|neg|positive|negative|positif|negatif|inc|ano|bl|date jour)$",
            desc,
            flags=re.I,
        )
        if m:
            desc = clean_text(m.group(1))
            result = clean_text(m.group(2))

    # Case 2: desc + numeric result merged
    if desc and not result:
        m = re.match(r"^(.*?)(?:\s+)([-+]?\d+(?:[.,]\d+)?[+-]?)$", desc)
        if m:
            left = clean_text(m.group(1))
            num = clean_text(m.group(2))
            if left:
                desc = left
                result = num

    # Case 3: result + unit merged
    if result and not unit:
        m = re.match(r"^([-+]?\d+(?:[.,]\d+)?[+-]?)\s*(.*)$", result)
        if m:
            maybe_num = clean_text(m.group(1))
            maybe_unit = clean_text(m.group(2))
            if maybe_unit:
                result = maybe_num
                unit = maybe_unit

    unit = unit.replace("g/di", "g/dl")
    unit = unit.replace("ux3", "μ×3")
    unit = unit.replace("UI/I37c", "UI/L37c")
    unit = unit.replace("UI/137c", "UI/L37c")
    unit = unit.replace("UI/I", "UI/L")
    unit = unit.replace("ui/l", "UI/L")
    unit = unit.replace("umol/i", "umol/L")
    unit = unit.replace("mml/l", "mmol/L")
    if unit == "g/":
        unit = "g/L"

    if val in JUNK_TOKENS:
        val = ""

    return [desc, result, unit, normal, val]


def fix_merged_result_unit_biochemistry(row):
    """Fix merged columns for biochemistry tables (5 columns: desc|result|unit|ref_min|ref_max)"""
    row = sanitize_columns(row)
    if len(row) < 5:
        row = row + [""] * (5 - len(row))

    desc, result, unit, ref_min, ref_max = row[:5]

    # Remove junk from ref columns
    if ref_max in JUNK_TOKENS:
        ref_max = ""

    # Normalize units
    unit = unit.replace("ui/l", "UI/L")
    unit = unit.replace("UI/137C", "UI/L37c")
    unit = unit.replace("UI/137c", "UI/L37c")
    unit = unit.replace("umol/l", "umol/L")
    unit = unit.replace("μmol/l", "umol/L")
    unit = unit.replace("g/dl", "g/L")
    unit = unit.replace("mmol/l", "mmol/L")

    return [desc, result, unit, ref_min, ref_max]


def parse_hybrid_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    section = None
    upper = []
    core = []
    lower = []

    for line in text:
        if line.startswith("=== UPPER SECTION"):
            section = "upper"
            continue
        if line.startswith("=== CORE TABLE"):
            section = "core"
            continue
        if line.startswith("=== LOWER SECTION"):
            section = "lower"
            continue

        if (
            line.startswith("PP box:")
            or line.startswith("TATR raw box:")
            or line.startswith("TATR scaled box:")
            or line.startswith("PP page size:")
            or line.startswith("TATR render size:")
            or line.startswith("OCR quality:")
            or line.startswith("HTML quality:")
            or line.startswith("Classic hematology page:")
            or line.startswith("Hematology OCR incomplete:")
            or line.startswith("Used HTML fallback:")
            or line.startswith("Detected section families:")
            or line.startswith("Biochemistry table page:")
            or line.startswith("  - ")
        ):
            continue

        line = line.rstrip()
        if not line:
            continue

        if section == "upper":
            upper.append(line)
        elif section == "core":
            core.append(line)
        elif section == "lower":
            lower.append(line)

    return upper, core, lower


def parse_pipe_row(line: str, is_biochemistry: bool = False):
    parts = [clean_text(x) for x in line.split("|")]
    
    if is_biochemistry:
        # Ensure 5 columns for biochemistry
        row = parts[:5] + [""] * max(0, 5 - len(parts[:5]))
        return fix_merged_result_unit_biochemistry(row[:5])
    else:
        # Ensure 5 columns for hematology
        row = parts[:5] + [""] * max(0, 5 - len(parts[:5]))
        return fix_merged_result_unit_hematology(row[:5])


def is_obvious_garbage_row(desc: str, result: str, unit: str, normal: str, val: str) -> bool:
    joined = norm(" ".join([desc, result, unit, normal, val]))

    garbage_patterns = [
        "lae",
        "tel:",
        "75010 paris 10",
        "75019 paris",
        "hopital robert debre",
        "consultit",
        "resultats d'une demande",
        "demande",
        "nom patient",
        "date naissance",
        "prescripteur",
        "patient adresse",
        "copie a",
        "echantillon",
        "prelevement",
        "requestresults.aspx",
        "aubervilliers",
    ]
    return any(p in joined for p in garbage_patterns)


def count_nonempty(row):
    return sum(1 for x in row if clean_text(x))


def has_numeric_value(text: str) -> bool:
    return bool(re.search(r"\d+[.,]\d+|\b\d+\b", text))


def has_known_unit(text: str) -> bool:
    return bool(
        re.search(
            r"%|g/dl|g/100ml|pg/hematie|10x3/mm3|10x6/mm3|/mm3|/mm³|μx3|ux3|μ×3|"
            r"mmhg|mmol/l|umol/l|ui/l|ml/100|g pour 100 ml|g/l",
            text,
            re.I,
        )
    )


def has_qualitative_value(text: str) -> bool:
    return bool(
        re.search(
            r"\b(non|oui|pos|neg|positive|negative|positif|negatif|ano|bl|inc|date jour)\b",
            text,
            re.I,
        )
    )


def is_known_biochem_analyte(text: str) -> bool:
    t = norm(clean_text(text))
    known = [
        "hemolyse",
        "lipemie",
        "sodium",
        "potassium",
        "chlore",
        "bicarbonates",
        "proteines plasmatiques",
        "uree",
        "creatinine",
        "glycemie",
        "phosphatases alcalines",
        "bilirubinetotale",
        "bilirubine totale",
        "bilirubine conjuguee",
        "asat",
        "alat",
        "ggt",
        "prealbumine",
    ]
    return any(k == t or k in t for k in known)


def row_has_table_shape(row):
    desc, result, unit, normal, val = row

    desc = clean_text(desc)
    result = clean_text(result)
    unit = clean_text(unit)
    normal = clean_text(normal)
    val = clean_text(val)

    nonempty = count_nonempty(row)

    if not desc:
        return False

    # keep known biochemistry analytes even if OCR missed the rest
    if is_known_biochem_analyte(desc):
        return True

    # Strongest case: description + some actual value-like content
    if result:
        if has_numeric_value(result) or has_qualitative_value(result):
            return True
        if unit or normal or val:
            return True

    # Description + unit/reference often still indicates a lab row
    if unit and (has_known_unit(unit) or has_known_unit(desc)):
        return True

    if normal and has_numeric_value(normal):
        return True

    # Multi-column structured row
    if nonempty >= 3:
        return True

    return False


def row_looks_like_note(row):
    desc, result, unit, normal, val = row
    joined = clean_text(" ".join([desc, result, unit, normal, val]))
    joined_norm = norm(joined)

    # sentence-like or comment-like text
    word_count = len(joined.split())
    long_sentence = word_count >= 6 and not unit and not normal

    # broken symbolic spill
    weird_symbols = bool(re.search(r"[一□■☑✔✗✘]", joined))

    # isolated narrative-style equality/comment line
    eq_comment = "=" in joined and not unit and not normal

    # if desc exists but row is weakly structured, treat as note
    weak_structure = not row_has_table_shape(row)

    # classic comment rows often have one long desc and almost no clean split
    if long_sentence:
        return True
    if weird_symbols and weak_structure:
        return True
    if eq_comment and weak_structure:
        return True

    return False


def classify_row(row):
    desc, result, unit, normal, val = row

    if is_obvious_garbage_row(desc, result, unit, normal, val):
        return "garbage"

    joined = clean_text(" ".join(row))
    if not joined:
        return "garbage"

    if is_metadata_line(joined) or is_header_line(joined):
        return "garbage"

    if is_section_title(joined):
        return "section"

    if row_looks_like_note(row):
        return "note"

    if row_has_table_shape(row):
        return "table"

    return "note"


def force_morphology_rows(table_rows, note_lines):
    def n(x):
        return " ".join(str(x).lower().split())

    # Look at all page content already collected by the cleaner
    all_text = " ".join(
        [" ".join([str(c) for c in row if clean_text(c)]) for row in table_rows] + note_lines
    )
    all_text_n = n(all_text)

    leuco_label_present = "anomalies morph leucocyte" in all_text_n or "anomaliesmorph leucocyte" in all_text_n
    plaq_label_present = "anomalies morph plaquette" in all_text_n or "anomaliesmorph plaquette" in all_text_n

    has_leuco = any(
        "anomalies morph leucocyte" in n(r[0]) or "anomaliesmorph leucocyte" in n(r[0])
        for r in table_rows
    )
    has_plaq = any(
        "anomalies morph plaquette" in n(r[0]) or "anomaliesmorph plaquette" in n(r[0])
        for r in table_rows
    )

    if leuco_label_present and not has_leuco:
        table_rows.append(["Anomalies morph leucocyte", "ANO", "", "", ""])

    if plaq_label_present and not has_plaq:
        table_rows.append(["Anomalies morph plaquette", "ANO", "", "", ""])

    cleaned_notes = []
    for line in note_lines:
        ln = n(line)

        if leuco_label_present and (
            "anomalies morph leucocyte" in ln or "anomaliesmorph leucocyte" in ln
        ):
            continue

        if plaq_label_present and (
            "anomalies morph plaquette" in ln or "anomaliesmorph plaquette" in ln
        ):
            continue

        cleaned_notes.append(line)

    return table_rows, cleaned_notes


def main():
    parser = argparse.ArgumentParser(description="Final cleanup for hybrid table parser output - supports both hematology and biochemistry")
    parser.add_argument("hybrid_txt", help="Path to hybrid parser txt file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables_final",
        help="Directory to save final cleaned output",
    )
    args = parser.parse_args()

    hybrid_txt = Path(args.hybrid_txt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    upper, core, lower = parse_hybrid_file(hybrid_txt)

    # Detect if this is a biochemistry table
    is_biochemistry = is_biochemistry_table_output(core)

    final_rows = []
    notes = []

    def handle_line_as_row(line: str):
        line = clean_text(line)
        if not line or line in JUNK_TOKENS:
            return

        if "|" in line:
            row = parse_pipe_row(line, is_biochemistry=is_biochemistry)
        else:
            if is_biochemistry:
                row = fix_merged_result_unit_biochemistry([line, "", "", "", ""])
            else:
                row = fix_merged_result_unit_hematology([line, "", "", "", ""])

        kind = classify_row(row)

        if kind == "garbage":
            return
        if kind == "section":
            notes.append(clean_text(" ".join([x for x in row if x])))
            return
        if kind == "note":
            notes.append(clean_text(" ".join([x for x in row if x])))
            return
        if kind == "table":
            final_rows.append(row)

    for line in upper:
        handle_line_as_row(line)

    for line in core:
        handle_line_as_row(line)

    for line in lower:
        handle_line_as_row(line)

    seen_rows = set()
    dedup_rows = []
    for row in final_rows:
        if is_biochemistry:
            row = fix_merged_result_unit_biochemistry(row)
        else:
            row = fix_merged_result_unit_hematology(row)
        key = tuple(row)
        if key not in seen_rows:
            seen_rows.add(key)
            dedup_rows.append(row)

    seen_notes = set()
    dedup_notes = []
    for n in notes:
        n = clean_text(n)
        if not n:
            continue
        if n not in seen_notes:
            seen_notes.add(n)
            dedup_notes.append(n)

    out_file = out_dir / f"{hybrid_txt.stem}_final.txt"
    
    # Only apply morphology forcing for hematology tables
    if not is_biochemistry:
        dedup_rows, dedup_notes = force_morphology_rows(dedup_rows, dedup_notes)
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== FINAL TABLE ===\n\n")
        
        if is_biochemistry:
            f.write("Description | Résultat | Unité | Valeurs normales (min) | Valeurs normales (max)\n")
        else:
            f.write("Description | Résultat | Unité | Valeurs normales | Val.\n")
        
        for row in dedup_rows:
            f.write(" | ".join(row) + "\n")

        if dedup_notes:
            f.write("\n=== FINAL NOTES / COMMENT SECTION ===\n\n")
            for n in dedup_notes:
                f.write(n + "\n")

    print(f"Saved final cleaned output: {out_file}")
    print(f"Table type: {'Biochemistry' if is_biochemistry else 'Hematology'}")
    print(f"Final table rows: {len(dedup_rows)}")
    print(f"Final note lines: {len(dedup_notes)}")


if __name__ == "__main__":
    main()
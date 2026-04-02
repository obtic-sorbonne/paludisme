from pathlib import Path
import argparse
import re


JUNK_TOKENS = {"回", "国", "□", "网", "\\", "……", ""}


def norm(s: str) -> str:
    return " ".join(
        s.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ï", "i")
        .replace("’", "'")
        .split()
    )


def clean_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def is_metadata_line(line: str) -> bool:
    t = norm(line)
    keys = [
        "nom patient",
        "date / heure",
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
        "75019 paris19",
    ]
    return any(k in t for k in keys)


def is_header_line(line: str) -> bool:
    t = norm(line)
    keys = ["description", "resultat", "unite", "valeursnormales", "valeurs normales", "val"]
    return sum(1 for k in keys if k in t) >= 2


def is_section_title(line: str) -> bool:
    t = norm(line)
    keys = [
        "examens d'hematologie",
        "cytologie",
    ]
    return any(k in t for k in keys)


def looks_like_lab_row_text(line: str) -> bool:
    t = clean_text(line)
    if not t:
        return False
    if is_metadata_line(t) or is_header_line(t):
        return False
    has_num = bool(re.search(r"\d+[.,]\d+|\b\d+\b", t))
    has_unit = bool(re.search(r"%|g/dl|g/100ml|pg/hematie|10x3/mm3|10X6/mm3|ux3|μ×3", t, re.I))
    has_alpha = bool(re.search(r"[A-Za-zÀ-ÿ]", t))
    return has_alpha and (has_num or has_unit)


def sanitize_columns(cols):
    out = []
    for c in cols:
        c = clean_text(c)
        if c in JUNK_TOKENS:
            c = ""
        out.append(c)
    return out


def fix_merged_result_unit(row):
    """
    Try to split rows like:
    ['Polybasophiles. 0,60', '', '%', '', '']
    into:
    ['Polybasophiles.', '0,60', '%', '', '']
    """
    row = sanitize_columns(row)
    if len(row) < 5:
        row = row + [""] * (5 - len(row))

    desc, result, unit, normal, val = row[:5]

    if not result:
        m = re.match(r"^(.*?)(\d+[.,]\d+)$", desc)
        if m:
            left = clean_text(m.group(1))
            num = clean_text(m.group(2))
            if left:
                desc = left
                result = num

    # normalize some OCR unit variants
    unit = unit.replace("g/di", "g/dl")
    unit = unit.replace("ux3", "μ×3")

    # drop junk val col
    if val in JUNK_TOKENS:
        val = ""

    return [desc, result, unit, normal, val]


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
        if line.startswith("PP box:") or line.startswith("TATR raw box:") or line.startswith("TATR scaled box:") or line.startswith("PP page size:") or line.startswith("TATR render size:"):
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


def main():
    parser = argparse.ArgumentParser(description="Final cleanup for hybrid table parser output")
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

    final_rows = []
    notes = []

    # Upper: keep only actual lab rows or section titles
    for line in upper:
        if is_metadata_line(line) or is_header_line(line):
            continue
        if is_section_title(line):
            notes.append(line)
            continue
        if looks_like_lab_row_text(line):
            parts = [clean_text(x) for x in line.split("|")]
            parts = [p for p in parts if p != ""]
            row = parts[:5] + [""] * max(0, 5 - len(parts[:5]))
            row = fix_merged_result_unit(row[:5])
            final_rows.append(row)

    # Core: keep rows as main table
    for line in core:
        parts = [clean_text(x) for x in line.split("|")]
        row = parts[:5] + [""] * max(0, 5 - len(parts[:5]))
        row = fix_merged_result_unit(row[:5])

        joined = " ".join(row).strip()
        if not joined:
            continue
        if is_metadata_line(joined) or is_header_line(joined):
            continue
        if looks_like_lab_row_text(joined):
            final_rows.append(row)

    # Lower: keep full note section, but make sure Rech parasites sang is included if present
    for line in lower:
        line = clean_text(line)
        if not line:
            continue
        if line in JUNK_TOKENS:
            continue
        notes.append(line)

    # de-duplicate rows while preserving order
    seen_rows = set()
    dedup_rows = []
    for row in final_rows:
        key = tuple(row)
        if key not in seen_rows:
            seen_rows.add(key)
            dedup_rows.append(row)

    # de-duplicate notes
    seen_notes = set()
    dedup_notes = []
    for n in notes:
        if n not in seen_notes:
            seen_notes.add(n)
            dedup_notes.append(n)

    out_file = out_dir / f"{hybrid_txt.stem}_final.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== FINAL TABLE ===\n\n")
        f.write("Description | Résultat | Unité | Valeurs normales | Val.\n")
        for row in dedup_rows:
            f.write(" | ".join(row) + "\n")

        if dedup_notes:
            f.write("\n=== FINAL NOTES / COMMENT SECTION ===\n\n")
            for n in dedup_notes:
                f.write(n + "\n")

    print(f"Saved final cleaned output: {out_file}")
    print(f"Final table rows: {len(dedup_rows)}")
    print(f"Final note lines: {len(dedup_notes)}")


if __name__ == "__main__":
    main()
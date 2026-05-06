#!/usr/bin/env python3
"""
fix_glm_tables.py
Location: ~/digitize_medical_records/SystemB_page_classification/fix_glm_tables.py

Fixes known GLM-OCR table parsing issues in clinical reports.
Specifically:
  1. Ionogramme sanguin - row labels missing
  2. Bilan hépatique - row labels missing

These tables always appear in the same order in Robert Debré hospital reports.
We detect the table by its header and inject the correct row labels.

Usage:
  python fix_glm_tables.py --file /path/to/page.txt
  python fix_glm_tables.py --dir /path/to/text_dir/
"""

import re
import argparse
from pathlib import Path

# ── Known table schemas ────────────────────────────────────────────────────────

# Ionogramme sanguin - always in this order
IONOGRAMME_ROWS = [
    "Sodium (mmol/l)",
    "Potassium (mmol/l)",
    "Chlore (mmol/l)",
    "Bicarbonates (mmol/l)",
    "Protéines (g/l)",
    "Urée (mmol/l)",
    "Créatinine (µmol/l)",
]

# Bilan hépatique - always in this order
BILAN_HEPATIQUE_ROWS = [
    "Phosphatases alcalines (UI/l)",
    "Bilirubine totale (µmol/l)",
    "Bilirubine conjuguée (µmol/l)",
    "ASAT (UI/l)",
    "ALAT (UI/l)",
    "Gamma GT (UI/l)",
    "Pré albumine (g/l)",
]


def is_data_row(line: str) -> bool:
    """Check if a markdown table row contains only numbers/empty cells."""
    line = line.strip()
    if not line.startswith("|"):
        return False
    # Remove leading/trailing pipes
    cells = [c.strip() for c in line.strip("|").split("|")]
    # A data row has numeric values (or empty) in all cells except maybe first
    numeric_pattern = re.compile(r'^[\d,.\s\-<>]+$|^$|^amas$|^ANO$|^BL$')
    return all(numeric_pattern.match(c) for c in cells if c != ":---")


def is_separator_row(line: str) -> bool:
    """Check if line is markdown table separator (|:---|:---|)"""
    return bool(re.match(r'\|[\s:|-]+\|', line.strip()))


def fix_ionogramme_table(text: str) -> str:
    """
    Fix ionogramme sanguin table by injecting row labels.
    
    Before:
      | Sodium (mmol/l) | 21.9.06 | 25.9.06 |
      | :--- | :--- | :--- |
      | 132 | 131 | |
      | 4,6 | 4 | |
      ...
    
    After:
      | Paramètre | 21.9.06 | 25.9.06 |
      | :--- | :--- | :--- |
      | Sodium (mmol/l) | 132 | 131 |
      | Potassium (mmol/l) | 4,6 | 4 |
      ...
    """
    # Pattern: ionogramme header line
    # GLM puts "Sodium (mmol/l)" in the header because it lost the first column
    pattern = re.compile(
        r'(\|\s*Sodium\s*\(mmol/l\)\s*\|[^\n]+\n'   # header with Sodium
        r'\|\s*:---[^\n]+\n'                           # separator
        r'(?:\|[^\n]+\n)*)',                           # data rows
        re.IGNORECASE
    )

    def replace_ionogramme(match):
        block = match.group(0)
        lines = block.strip().split('\n')

        # Extract date headers from first line
        header_line = lines[0]
        # Get the date columns (skip the "Sodium" part which is actually a row label)
        cells = [c.strip() for c in header_line.strip('|').split('|')]
        # cells[0] = "Sodium (mmol/l)" → this was the row label, dates follow
        date_cols = cells[1:]  # e.g. ["21.9.06", "25.9.06"]

        # Build new header
        new_header = "| Paramètre | " + " | ".join(date_cols) + " |"
        new_sep = "| :--- | " + " | ".join([":---"] * len(date_cols)) + " |"

        # Extract data rows (skip header and separator)
        data_lines = [l for l in lines[2:] if l.strip() and is_data_row(l)]

        # Build new rows with labels injected
        new_rows = []
        for i, data_line in enumerate(data_lines):
            if i < len(IONOGRAMME_ROWS):
                label = IONOGRAMME_ROWS[i]
                # Extract values from data row
                vals = [c.strip() for c in data_line.strip('|').split('|')]
                new_row = f"| {label} | " + " | ".join(vals) + " |"
                new_rows.append(new_row)
            else:
                new_rows.append(data_line)

        result = "\n".join([new_header, new_sep] + new_rows) + "\n"
        return result

    return pattern.sub(replace_ionogramme, text)


def fix_bilan_hepatique_table(text: str) -> str:
    """
    Fix bilan hépatique table by injecting row labels.
    The bilan hépatique header has only dates (no label column at all).
    
    Before:
      | 21.9.06 | 24.9.06 | 25.9.06 |
      | :--- | :--- | :--- |
      | 153 | 82 | 100 |
      ...
    
    After:
      | Paramètre | 21.9.06 | 24.9.06 | 25.9.06 |
      | :--- | :--- | :--- | :--- |
      | Phosphatases alcalines (UI/l) | 153 | 82 | 100 |
      ...
    """
    # Find "Bilan hépatique" marker then the table that follows
    pattern = re.compile(
        r'(\*\s*Bilan h[eé]patique\s*:\s*\n'          # "* Bilan hépatique :"
        r'(\|\s*[\d.]+[^\n]+\n'                        # header with dates only
        r'\|\s*:---[^\n]+\n'                           # separator
        r'(?:\|[^\n]+\n)*))',                           # data rows
        re.IGNORECASE
    )

    def replace_bilan(match):
        full_match = match.group(1)
        # Split off the "* Bilan hépatique :" line
        first_newline = full_match.index('\n')
        prefix = full_match[:first_newline + 1]
        table_part = full_match[first_newline + 1:]

        lines = table_part.strip().split('\n')
        if not lines:
            return full_match

        # Extract date columns from header
        header_line = lines[0]
        date_cols = [c.strip() for c in header_line.strip('|').split('|')]

        # Build new header with Paramètre column
        new_header = "| Paramètre | " + " | ".join(date_cols) + " |"
        new_sep = "| :--- | " + " | ".join([":---"] * len(date_cols)) + " |"

        # Extract data rows
        data_lines = [l for l in lines[2:] if l.strip() and is_data_row(l)]

        # Build new rows with labels
        new_rows = []
        for i, data_line in enumerate(data_lines):
            if i < len(BILAN_HEPATIQUE_ROWS):
                label = BILAN_HEPATIQUE_ROWS[i]
                vals = [c.strip() for c in data_line.strip('|').split('|')]
                new_row = f"| {label} | " + " | ".join(vals) + " |"
                new_rows.append(new_row)
            else:
                new_rows.append(data_line)

        new_table = "\n".join([new_header, new_sep] + new_rows) + "\n"
        return prefix + new_table

    return pattern.sub(replace_bilan, text)


def fix_text(text: str) -> str:
    """Apply all table fixes to a text."""
    text = fix_ionogramme_table(text)
    text = fix_bilan_hepatique_table(text)
    return text


def fix_file(path: Path, inplace: bool = True) -> bool:
    """Fix tables in one file. Returns True if any changes made."""
    original = path.read_text(encoding="utf-8", errors="replace")
    fixed = fix_text(original)

    if fixed == original:
        return False

    if inplace:
        path.write_text(fixed, encoding="utf-8")
    else:
        out_path = path.with_stem(path.stem + "_fixed")
        out_path.write_text(fixed, encoding="utf-8")
        print(f"  Saved → {out_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix GLM-OCR table parsing for clinical reports"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Fix one text file")
    group.add_argument("--dir",  help="Fix all .txt files in directory")
    parser.add_argument("--inplace", action="store_true", default=True,
                        help="Edit files in place (default: True)")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        changed = fix_file(path, inplace=args.inplace)
        print(f"{'✅ Fixed' if changed else '  No change'}: {path}")

    else:
        txt_files = sorted(Path(args.dir).glob("*.txt"))
        changed_count = 0
        for path in txt_files:
            changed = fix_file(path, inplace=args.inplace)
            if changed:
                print(f"  ✅ Fixed: {path.name}")
                changed_count += 1
        print(f"\nFixed {changed_count}/{len(txt_files)} files")


if __name__ == "__main__":
    main()
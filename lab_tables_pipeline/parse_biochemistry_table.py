"""
Dedicated parser for biochemistry lab tables from French medical documents.
Handles the specific layout and OCR challenges of Hôpital Robert Debré biochemistry pages.

Key characteristics:
- 5 columns: Description | Résultat | Unité | Valeurs normales (min) | Valeurs normales (max)
- Qualitative rows: Hemolyse, Lipemie (no unit, no reference)
- Quantitative rows: All others with result, unit, and reference range
- OCR issue: Reference ranges get grouped with the NEXT row due to Y-coordinate overlap
- Solution: Detect missing references and recover them from the next row
"""

import re
from pathlib import Path
from collections import defaultdict


def clean_text(text):
    """Clean and normalize text"""
    text = str(text).strip()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def norm(s: str) -> str:
    """Normalize text for comparison"""
    return " ".join(
        s.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ï", "i")
        .replace("î", "i")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("'", "'")
        .split()
    )


def looks_like_metadata_text(text):
    """Check if text is metadata/header"""
    t = norm(text)
    keys = [
        "nom patient", "date / heure", "date naissance", "prescripteur",
        "adresse", "patient adresse", "copie a", "echantillon", "prelevement",
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2", "page 2 sur 2",
        "hopital robert debre", "urgent"
    ]
    return any(k in t for k in keys)


def looks_like_header_text(text):
    """Check if text is column header"""
    t = norm(text)
    keys = ["description", "resultat", "unite", "valeurs normales", "val"]
    return sum(1 for k in keys if k in t) >= 2


def is_noise_text(text):
    """Check if text is junk/noise"""
    t = text.strip()
    return (not t) or t in {
        "网", "回", "国", "\\", "……", "□", "■", "☑", "√", "✔", "✗", "✘", "区", "日"
    }


def looks_like_biochem_result(text):
    """Detect result values: qualitative or numeric"""
    t = clean_text(text)
    return bool(re.fullmatch(
        r"(non|oui|pos|neg|positive|negative|positif|negatif|[-+]?\d+(?:[.,]\d+)?[+-]?)",
        t,
        re.I
    ))


def looks_like_biochem_unit(text):
    """Detect biochemistry units"""
    t = norm(clean_text(text))
    return bool(re.search(
        r"\b(mmol/l|umol/l|μmol/l|g/l|ui/l|ui/l37c|ui/137c|ui/l 37c|l37c)\b",
        t,
        re.I
    ))


def looks_like_biochem_reference(text):
    """Detect reference ranges: '134 142', '3,1 4,7', '0 17', '120 400'"""
    t = clean_text(text)
    return bool(re.search(r"[-+]?\d+[.,]?\d*\s+[-+]?\d+[.,]?\d*", t))


def normalize_biochem_unit(text):
    """Normalize unit notation"""
    t = clean_text(text)
    t = t.replace("ui/l", "UI/L")
    t = t.replace("UI/137C", "UI/L37c")
    t = t.replace("UI/137c", "UI/L37c")
    t = t.replace("UI/137℃", "UI/L37c")
    t = t.replace("UI/L37C", "UI/L37c")
    t = t.replace("umol/l", "umol/L")
    t = t.replace("μmol/l", "umol/L")
    t = t.replace("mmol/l", "mmol/L")
    t = t.replace("g/l", "g/L")
    return t


def assign_cols(row, bounds):
    """Assign items to columns based on boundaries"""
    if bounds is None:
        cols = [""] * 5
        sorted_items = sorted(row, key=lambda x: x["x1"])
        for i, item in enumerate(sorted_items[:5]):
            cols[i] = item["text"].strip()
        return cols
    
    cols = ["", "", "", "", ""]
    for item in row:
        x = item["cx"]
        text = item["text"].strip()
        
        idx = 0
        for i, bound in enumerate(bounds):
            if x >= bound:
                idx = i + 1
        idx = min(idx, 4)
        
        cols[idx] = (cols[idx] + " " + text).strip() if cols[idx] else text
    
    return cols


def cleanup_bad_rows(rows):
    """Remove completely empty or metadata rows"""
    cleaned = []
    for row in rows:
        row = [clean_text(c) for c in row]
        if all(not c for c in row):
            continue
        joined = " ".join(row)
        if looks_like_metadata_text(joined):
            continue
        cleaned.append(row)
    return cleaned


def parse_biochemistry_table(all_rows, tatr_box):
    """
    Parse biochemistry table with comprehensive recovery strategy.
    
    Strategy:
    1. Assign columns using fixed percentages (biochemistry-specific)
    2. First pass: Raw column assignment
    3. Second pass: Detect and merge orphan reference rows
    4. Third pass: Assign column roles (description, result, unit, reference)
    5. Fourth pass: Recovery - shift missing references from next row
    
    Args:
        all_rows: List of row objects with x1, y1, x2, y2, cx, cy, text
        tatr_box: [x1, y1, x2, y2] table bounding box
    
    Returns:
        List of [description, result, unit, reference_min, reference_max] rows
    """
    
    # Fixed percentages for biochemistry layout
    x1, _, x2, _ = tatr_box
    w = x2 - x1
    bounds = [
        x1 + 0.35 * w,    # Column 1: Description (0-35%)
        x1 + 0.50 * w,    # Column 2: Result (35-50%)
        x1 + 0.70 * w,    # Column 3: Unit (50-70%)
        x1 + 0.90 * w,    # Column 4: Reference (70-90%)
    ]

    # ============================================================================
    # PASS 1: Raw column assignment
    # ============================================================================
    raw_rows = []
    for row in all_rows:
        cols = [clean_text(c) for c in assign_cols(row, bounds)]
        joined = " ".join(cols).strip()

        if not joined:
            continue
        if looks_like_metadata_text(joined):
            continue
        if looks_like_header_text(joined):
            continue

        raw_rows.append(cols)

    # ============================================================================
    # PASS 2: Detect and merge orphan reference rows
    # ============================================================================
    def is_orphan_reference_row(cols):
        """Check if this row is just reference values with no description/result/unit"""
        desc, c2, c3, c4, c5 = cols
        
        has_desc = bool(clean_text(desc))
        has_result = bool(clean_text(c2))
        has_unit = bool(clean_text(c3))
        has_ref = bool(clean_text(c4) or clean_text(c5))
        
        return (not has_desc and not has_result and not has_unit and has_ref)

    merged_rows = []
    for cols in raw_rows:
        cols = [clean_text(c) for c in cols]
        
        if is_orphan_reference_row(cols):
            # Merge this orphan reference into the previous row
            if merged_rows:
                prev_row = merged_rows[-1]
                desc, result, unit, ref, val = prev_row
                _, _, _, orphan_ref, _ = cols
                
                if orphan_ref and not ref:
                    ref = orphan_ref
                    merged_rows[-1] = [desc, result, unit, ref, val]
            continue
        
        merged_rows.append(cols)

    # ============================================================================
    # PASS 3: Assign column roles
    # ============================================================================
    structured_rows = []
    
    for cols in merged_rows:
        desc, c2, c3, c4, c5 = cols

        joined = clean_text(" ".join(cols))
        if not joined:
            continue
        if looks_like_header_text(joined):
            continue

        # Clean junk tokens
        if c5 in {"国", "回", "□", "√", "■", "区", "日", "网", "☑", "✔", "✗", "✘"}:
            c5 = ""

        result = ""
        unit = ""
        ref = ""

        # Multi-pass token classification
        # Pass A: Simple classification for well-separated columns
        for token in [c2, c3, c4, c5]:
            token = clean_text(token)
            if not token:
                continue

            if not result and looks_like_biochem_result(token):
                result = token
                continue

            if not unit and looks_like_biochem_unit(token):
                unit = normalize_biochem_unit(token)
                continue

            if not ref and looks_like_biochem_reference(token):
                ref = token
                continue

        # Pass B: Handle result+unit merged in c2 (e.g., "132-" or "0,06 -")
        if not result and c2:
            m = re.match(r"^([-+]?\d+[.,]?\d*[\+\-]?)\s*(.*)$", c2)
            if m:
                numeric_part = clean_text(m.group(1))
                trailing = clean_text(m.group(2))
                
                if numeric_part:
                    result = numeric_part
                    if trailing and trailing not in {"-", "+"}:
                        if looks_like_biochem_unit(trailing):
                            unit = normalize_biochem_unit(trailing)

        # Pass C: If still no unit, check c3
        if not unit and c3 and looks_like_biochem_unit(c3):
            unit = normalize_biochem_unit(c3)

        # Pass D: If still no ref, check remaining columns
        if not ref:
            for token in [c4, c5]:
                token = clean_text(token)
                if token and looks_like_biochem_reference(token):
                    ref = token
                    break

        # Pass E: Special case for qualitative results (non/oui/pos/neg)
        if not result and c2 and not unit and not ref:
            if c2.lower() in {"non", "oui", "pos", "neg"}:
                result = c2

        structured_rows.append([desc, result, unit, ref, ""])

    # ============================================================================
    # PASS 4: Recovery - Shift missing references from next row
    # ============================================================================
    # This handles the case where references got grouped with the next analyte
    for i in range(1, len(structured_rows)):
        prev_row = structured_rows[i - 1]
        curr_row = structured_rows[i]
        
        prev_desc, prev_result, prev_unit, prev_ref, prev_val = prev_row
        curr_desc, curr_result, curr_unit, curr_ref, curr_val = curr_row
        
        # Conditions for shifting:
        # 1. Previous row has NO reference (this is the problem)
        # 2. Current row HAS a reference
        # 3. Both rows have results (ensures we're not dealing with garbage)
        # 4. Previous row has a result (it's a complete analyte except for ref)
        if (not prev_ref and curr_ref and prev_result and curr_result):
            # Move current row's reference to previous row
            structured_rows[i - 1] = [prev_desc, prev_result, prev_unit, curr_ref, prev_val]
            # Clear current row's reference (it wasn't supposed to be there)
            structured_rows[i] = [curr_desc, curr_result, curr_unit, "", curr_val]

    # ============================================================================
    # Final cleanup
    # ============================================================================
    structured_rows = cleanup_bad_rows(structured_rows)
    
    return structured_rows


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("Biochemistry Table Parser")
    print("=" * 80)
    print(__doc__)
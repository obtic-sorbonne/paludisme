from __future__ import annotations

import re

from narrative_table_utils import (
    clean_line,
    normalize_numeric_token,
    is_date_token,
    normalize_date_token,
    looks_like_section_heading,
)


def is_junk_footer_line(s: str) -> bool:
    t = clean_line(s).lower()
    bad = [
        "http://",
        "https://",
        "imprime le",
        "page 1/2",
        "page 2/2",
        "page 1 sur 2",
        "page 2 sur 2",
    ]
    return any(x in t for x in bad)


_MEASUREMENT_UNIT_PAT = re.compile(
    r"\(%\)|\(UI|\(/mm|\(g/|\(mmol|\(\xb5|\(\u03bc|\(10|\(pg|\(dl|\(l\)|\(ml\)",
    re.IGNORECASE,
)


def token_is_small_noise(tok_text: str) -> bool:
    t = clean_line(tok_text)
    if not t:
        return True
    if is_junk_footer_line(t):
        return True
    if t in {".", ",", ";", ":", "-", "--"}:
        return True
    return False


def cleanup_label(label: str) -> str:
    label = clean_line(label)
    label = re.sub(r"[.\xb7\x95]{2,}$", "", label).strip()
    label = re.sub(r"\s{2,}", " ", label)
    return label


def cleanup_value(value: str) -> str:
    value = normalize_numeric_token(value)
    value = re.sub(r"\s{2,}", " ", value)
    return value


def tokens_to_text(tokens) -> str:
    if not tokens:
        return ""
    tokens = sorted(tokens, key=lambda t: t["x1"])
    return clean_line(" ".join(tok["text"] for tok in tokens))


def row_text(row) -> str:
    return clean_line(" ".join(tok["text"] for tok in row))


def group_tokens_into_rows(tokens, y_tol=9):
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t["cy"], t["x1"]))
    rows = []
    current = [toks[0]]
    for tok in toks[1:]:
        avg_y = sum(x["cy"] for x in current) / len(current)
        if abs(tok["cy"] - avg_y) <= y_tol:
            current.append(tok)
        else:
            rows.append(sorted(current, key=lambda x: x["x1"]))
            current = [tok]
    if current:
        rows.append(sorted(current, key=lambda x: x["x1"]))
    return rows


def detect_date_header_row(rows):
    best_idx, best_toks = None, []
    for idx, row in enumerate(rows):
        date_toks = [t for t in row if is_date_token(t["text"])]
        if len(date_toks) > len(best_toks):
            best_idx, best_toks = idx, date_toks
    if len(best_toks) >= 2:
        return best_idx, sorted(best_toks, key=lambda t: t["cx"])
    return None, []


def build_column_boundaries_from_dates(date_tokens, box_left, box_right):
    date_tokens = sorted(date_tokens, key=lambda t: t["cx"])
    centers = [t["cx"] for t in date_tokens]
    dates = [normalize_date_token(t["text"]) for t in date_tokens]

    midpoints = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]

    if len(centers) >= 2:
        last_half_gap = (centers[-1] - centers[-2]) / 2.0
    else:
        last_half_gap = 40.0

    first_value_left = max(box_left, min(t["x1"] for t in date_tokens))
    last_value_right = min(box_right, centers[-1] + last_half_gap)

    boundaries = [first_value_left] + midpoints + [last_value_right]
    value_intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(dates))]
    label_right = first_value_left - 5

    return {
        "dates": dates,
        "centers": centers,
        "value_intervals": value_intervals,
        "label_right": label_right,
    }


def assign_token_to_column(tok, value_intervals):
    cx = tok["cx"]
    for col_idx, (xl, xr) in enumerate(value_intervals):
        if xl <= cx < xr:
            return col_idx

    best_col, best_dist = None, float("inf")
    for col_idx, (xl, xr) in enumerate(value_intervals):
        centre = (xl + xr) / 2.0
        dist = abs(cx - centre)
        col_width = xr - xl
        if dist < col_width * 0.6 and dist < best_dist:
            best_dist = dist
            best_col = col_idx
    return best_col


def build_label_candidates(tokens_in_box, label_right):
    label_tokens = [
        t
        for t in tokens_in_box
        if t["cx"] < label_right + 8 and not token_is_small_noise(t["text"])
    ]
    if not label_tokens:
        return []

    label_rows = group_tokens_into_rows(label_tokens, y_tol=8)
    candidates = []
    for row in label_rows:
        label = cleanup_label(tokens_to_text(row))
        if not label or is_date_token(label):
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", label):
            continue
        candidates.append(
            {
                "label": label,
                "y": sum(t["cy"] for t in row) / len(row),
                "tokens": row,
            }
        )
    return candidates


def build_value_entries(tokens_in_box, value_intervals, label_right):
    entries = []
    for col_idx, (x_left, x_right) in enumerate(value_intervals):
        col_tokens = [
            t
            for t in tokens_in_box
            if t["cx"] >= label_right and x_left <= t["cx"] < x_right and not token_is_small_noise(t["text"])
        ]
        if not col_tokens:
            continue

        col_rows = group_tokens_into_rows(col_tokens, y_tol=8)
        for row in col_rows:
            txt = cleanup_value(tokens_to_text(row))
            if not txt or is_date_token(txt):
                continue
            entries.append(
                {
                    "col_idx": col_idx,
                    "text": txt,
                    "y": sum(t["cy"] for t in row) / len(row),
                    "tokens": row,
                }
            )
    return entries


def assign_values_to_labels(label_candidates, value_entries, max_y_gap=40):
    if not label_candidates or not value_entries:
        return [], value_entries

    label_candidates = sorted(label_candidates, key=lambda x: x["y"])
    value_entries = sorted(value_entries, key=lambda x: x["y"])

    slot_best = {}
    unassigned = []

    for val in value_entries:
        gaps = sorted((abs(val["y"] - lab["y"]), i) for i, lab in enumerate(label_candidates))
        best_gap, best_idx = gaps[0]

        if best_gap > max_y_gap:
            unassigned.append(val)
            continue

        key = (best_idx, val["col_idx"])
        prev = slot_best.get(key)
        if prev is None or best_gap < prev[0]:
            slot_best[key] = (best_gap, val["text"])

    assigned = {i: {} for i in range(len(label_candidates))}
    for (label_idx, col_idx), (_, text) in slot_best.items():
        assigned[label_idx][col_idx] = text

    parsed_rows = []
    for i, lab in enumerate(label_candidates):
        vals = assigned.get(i, {})
        if vals:
            parsed_rows.append({"label": lab["label"], "values": vals})

    return parsed_rows, unassigned


def assign_values_monotonic_by_column(label_candidates, value_entries, max_y_gap=18):
    if not label_candidates:
        return [], value_entries

    label_candidates = sorted(label_candidates, key=lambda x: x["y"])
    n_labels = len(label_candidates)

    assigned = {i: {} for i in range(n_labels)}
    used_value_indices = set()

    cols = sorted(set(v["col_idx"] for v in value_entries))

    for col_idx in cols:
        col_vals = [(i, v) for i, v in enumerate(value_entries) if v["col_idx"] == col_idx]
        col_vals = sorted(col_vals, key=lambda x: x[1]["y"])

        v_ptr = 0
        for lab_idx, lab in enumerate(label_candidates):
            best_i = None
            best_gap = float("inf")

            scan_ptr = v_ptr
            while scan_ptr < len(col_vals):
                original_idx, ve = col_vals[scan_ptr]
                gap = ve["y"] - lab["y"]

                if gap < -6:
                    scan_ptr += 1
                    v_ptr = scan_ptr
                    continue

                abs_gap = abs(gap)
                if abs_gap <= max_y_gap and abs_gap < best_gap:
                    best_gap = abs_gap
                    best_i = scan_ptr

                if gap > max_y_gap:
                    break

                scan_ptr += 1

            if best_i is not None:
                original_idx, ve = col_vals[best_i]
                assigned[lab_idx][col_idx] = ve["text"]
                used_value_indices.add(original_idx)
                v_ptr = best_i + 1

    parsed_rows = []
    for i, lab in enumerate(label_candidates):
        vals = assigned.get(i, {})
        if vals:
            parsed_rows.append({"label": lab["label"], "values": vals})

    unassigned = [ve for i, ve in enumerate(value_entries) if i not in used_value_indices]
    return parsed_rows, unassigned


def parse_table_box_rowwise(tokens_in_box):
    rows = group_tokens_into_rows(tokens_in_box, y_tol=7)
    if len(rows) < 3:
        return None

    header_idx, header_date_tokens = detect_date_header_row(rows)
    if header_idx is None:
        return None

    box_left = min(t["x1"] for t in tokens_in_box) - 5
    box_right = max(t["x2"] for t in tokens_in_box) + 5
    structure = build_column_boundaries_from_dates(header_date_tokens, box_left, box_right)

    label_right = structure["label_right"]
    value_intervals = structure["value_intervals"]
    dates = structure["dates"]
    n_cols = len(dates)

    parsed_rows = []
    notes = []

    header_row = rows[header_idx]
    header_label_toks = [
        t
        for t in header_row
        if t["cx"] < label_right + 8 and not token_is_small_noise(t["text"]) and not is_date_token(t["text"])
    ]
    embedded_label = cleanup_label(tokens_to_text(header_label_toks))
    start_idx = header_idx + 1

    if embedded_label and re.search(r"[A-Za-zÀ-ÿ]", embedded_label):
        if start_idx < len(rows):
            next_row = rows[start_idx]
            value_toks = [
                t for t in next_row if t["cx"] >= label_right + 8 and not token_is_small_noise(t["text"])
            ]
            col_values = {}
            for tok in value_toks:
                col = assign_token_to_column(tok, value_intervals)
                if col is not None and col < n_cols:
                    val = cleanup_value(tok["text"])
                    if val and col not in col_values:
                        col_values[col] = val
            if col_values:
                parsed_rows.append({"label": embedded_label, "values": col_values})
            start_idx += 1

    for row in rows[start_idx:]:
        label_toks = [
            t for t in row if t["cx"] < label_right + 8 and not token_is_small_noise(t["text"])
        ]
        value_toks = [
            t for t in row if t["cx"] >= label_right + 8 and not token_is_small_noise(t["text"])
        ]

        label = cleanup_label(tokens_to_text(label_toks))
        if is_date_token(label) or is_junk_footer_line(label):
            continue

        col_values = {}
        unmatched = []
        for tok in value_toks:
            col = assign_token_to_column(tok, value_intervals)
            if col is not None and col < n_cols:
                val = cleanup_value(tok["text"])
                if val:
                    if col not in col_values:
                        col_values[col] = val
                    else:
                        col_values[col] = col_values[col] + " " + val
            else:
                unmatched.append(tok)

        if not label and not col_values:
            continue

        if label and not re.search(r"[A-Za-zÀ-ÿ]", label):
            orphan_vals = {}
            for tok in label_toks + value_toks:
                col = assign_token_to_column(tok, value_intervals)
                if col is not None and col < n_cols:
                    val = cleanup_value(tok["text"])
                    if val:
                        orphan_vals[col] = val
                else:
                    notes.append(cleanup_value(tok["text"]))
            if orphan_vals and parsed_rows:
                for col, val in orphan_vals.items():
                    if col not in parsed_rows[-1]["values"]:
                        parsed_rows[-1]["values"][col] = val
            else:
                for v in orphan_vals.values():
                    notes.append(v)
            continue

        if col_values:
            if label:
                parsed_rows.append({"label": label, "values": col_values})
            elif parsed_rows:
                for col, val in col_values.items():
                    if col not in parsed_rows[-1]["values"]:
                        parsed_rows[-1]["values"][col] = val
            else:
                for v in col_values.values():
                    notes.append(v)
        elif label:
            notes.append(label)

        for tok in unmatched:
            notes.append(cleanup_value(tok["text"]))

    if not parsed_rows:
        return None

    cleaned = []
    for row in parsed_rows:
        lbl = row["label"]
        if not lbl or is_junk_footer_line(lbl):
            continue
        if looks_like_section_heading(lbl) and len(lbl.split()) <= 3:
            continue
        cleaned.append(row)

    if not cleaned:
        return None

    return {"dates": dates, "rows": cleaned, "notes": notes}


def build_review_info(parsed_table, label_candidates, value_entries):
    reasons = []

    row_labels = [clean_line(r.get("label", "")) for r in parsed_table.get("rows", [])]
    notes = parsed_table.get("notes", [])

    if label_candidates and parsed_table.get("rows"):
        if len(parsed_table["rows"]) < max(3, len(label_candidates) - 2):
            reasons.append(
                f"Parsed rows ({len(parsed_table['rows'])}) fewer than detected label candidates ({len(label_candidates)})"
            )

    if len(notes) >= 3:
        reasons.append(f"{len(notes)} unassigned values/notes remain")

    if label_candidates and row_labels:
        first_candidate = clean_line(label_candidates[0]["label"])
        if first_candidate and first_candidate != row_labels[0]:
            reasons.append(f"First detected label candidate missing from parsed rows: {first_candidate}")

    joined_rows = " ".join(row_labels).lower()
    if "erythrocytes" not in joined_rows and any("erythrocytes" in clean_line(c["label"]).lower() for c in label_candidates):
        reasons.append("Erythrocytes row missing from final parsed rows")

    review_needed = len(reasons) > 0
    return review_needed, reasons


def parse_table_box_by_alignment(tokens_in_box):
    rows = group_tokens_into_rows(tokens_in_box, y_tol=10)
    if len(rows) < 3:
        return None

    header_idx, header_date_tokens = detect_date_header_row(rows)
    if header_idx is None:
        return None

    box_left = min(t["x1"] for t in tokens_in_box) - 5
    box_right = max(t["x2"] for t in tokens_in_box) + 5
    structure = build_column_boundaries_from_dates(header_date_tokens, box_left, box_right)

    label_right = structure["label_right"]
    value_intervals = structure["value_intervals"]
    dates = structure["dates"]

    parsed_rows = []
    notes = []

    header_row = rows[header_idx]
    header_label_toks = [
        t
        for t in header_row
        if t["cx"] < label_right + 8
        and not token_is_small_noise(t["text"])
        and not is_date_token(t["text"])
    ]
    embedded_label = cleanup_label(tokens_to_text(header_label_toks))

    start_idx = header_idx + 1
    embedded_used_cols = set()

    if embedded_label and re.search(r"[A-Za-zÀ-ÿ]", embedded_label):
        if start_idx < len(rows):
            next_row = rows[start_idx]
            value_toks = [
                t
                for t in next_row
                if t["cx"] >= label_right + 8
                and not token_is_small_noise(t["text"])
            ]

            col_values = {}
            for tok in value_toks:
                col = assign_token_to_column(tok, value_intervals)
                if col is not None:
                    val = cleanup_value(tok["text"])
                    if val and col not in col_values:
                        col_values[col] = val
                        embedded_used_cols.add(col)

            if col_values:
                parsed_rows.append({"label": embedded_label, "values": col_values})
            start_idx += 1

    header_bottom = max(t["y2"] for t in header_date_tokens)
    body_tokens = [t for t in tokens_in_box if t["cy"] > header_bottom - 2]

    if not body_tokens:
        return None

    label_candidates = build_label_candidates(body_tokens, label_right)

    if embedded_label:
        emb_norm = clean_line(embedded_label).lower()
        label_candidates = [
            lc for lc in label_candidates
            if clean_line(lc["label"]).lower() != emb_norm
        ]

    if not label_candidates and not parsed_rows:
        return None

    value_entries = build_value_entries(
        body_tokens,
        structure["value_intervals"],
        label_right,
    )

    if embedded_used_cols and start_idx < len(rows):
        filtered_value_entries = []
        next_row_y = sum(t["cy"] for t in rows[header_idx + 1]) / len(rows[header_idx + 1])
        for ve in value_entries:
            if ve["col_idx"] in embedded_used_cols and abs(ve["y"] - next_row_y) <= 8:
                continue
            filtered_value_entries.append(ve)
        value_entries = filtered_value_entries

    aligned_rows, unassigned_values = assign_values_monotonic_by_column(
        label_candidates,
        value_entries,
        max_y_gap=18,
    )

    all_rows = parsed_rows + aligned_rows
    if not all_rows:
        return None

    cleaned_rows = []
    for row in all_rows:
        label = cleanup_label(row["label"])
        if not label or is_junk_footer_line(label):
            continue
        if looks_like_section_heading(label) and len(label.split()) <= 3:
            continue
        cleaned_rows.append({"label": label, "values": row["values"]})

    if not cleaned_rows:
        return None

    seen = set()
    for ve in unassigned_values:
        txt = clean_line(ve["text"])
        if txt and txt not in seen and not is_junk_footer_line(txt):
            seen.add(txt)
            notes.append(txt)

    result = {
        "dates": dates,
        "rows": cleaned_rows,
        "notes": notes,
    }

    review_needed, review_reasons = build_review_info(result, label_candidates, value_entries)
    result["review_needed"] = review_needed
    result["review_reasons"] = review_reasons
    return result


def parse_table_box_generic(tokens_in_box):
    result = parse_table_box_by_alignment(tokens_in_box)
    if result:
        return result

    result = parse_table_box_rowwise(tokens_in_box)
    if result:
        result["review_needed"] = bool(result.get("notes"))
        result["review_reasons"] = ["Used rowwise fallback parser"]
        return result

    rows = group_tokens_into_rows(tokens_in_box, y_tol=10)
    if len(rows) < 3:
        return None

    header_idx, header_date_tokens = detect_date_header_row(rows)
    if header_idx is None:
        return None

    box_left = min(t["x1"] for t in tokens_in_box) - 5
    box_right = max(t["x2"] for t in tokens_in_box) + 5
    structure = build_column_boundaries_from_dates(header_date_tokens, box_left, box_right)
    label_right = structure["label_right"]

    header_bottom = max(t["y2"] for t in header_date_tokens)
    body_tokens = [t for t in tokens_in_box if t["cy"] > header_bottom - 2]
    if not body_tokens:
        return None

    label_candidates = build_label_candidates(body_tokens, label_right)
    if not label_candidates:
        return None

    value_entries = build_value_entries(body_tokens, structure["value_intervals"], label_right)
    parsed_rows, unassigned_values = assign_values_to_labels(label_candidates, value_entries, max_y_gap=40)
    if not parsed_rows:
        return None

    cleaned_rows = []
    for row in parsed_rows:
        label = cleanup_label(row["label"])
        if not label or is_junk_footer_line(label):
            continue
        if looks_like_section_heading(label) and len(label.split()) <= 3:
            continue
        cleaned_rows.append({"label": label, "values": row["values"]})

    if not cleaned_rows:
        return None

    notes = []
    seen = set()
    for val in unassigned_values:
        txt = clean_line(val["text"])
        if txt and txt not in seen and not is_junk_footer_line(txt):
            seen.add(txt)
            notes.append(txt)

    result = {"dates": structure["dates"], "rows": cleaned_rows, "notes": notes}
    result["review_needed"] = True
    result["review_reasons"] = ["Used generic Y-proximity fallback parser"]
    return result


def format_parsed_table(title_lines, parsed_table):
    out = []

    seen_titles = set()
    for t in title_lines:
        tt = clean_line(t)
        if not tt:
            continue
        skip_prefixes = ("nda :", "npi :", "ne(e) le", "patient :")
        if any(tt.lower().startswith(p) for p in skip_prefixes):
            continue
        if tt not in seen_titles:
            out.append(tt)
            seen_titles.add(tt)

    if out:
        out.append("")

    dates = parsed_table.get("dates", [])
    rows = parsed_table.get("rows", [])
    notes = parsed_table.get("notes", [])
    review_needed = parsed_table.get("review_needed", False)
    review_reasons = parsed_table.get("review_reasons", [])

    for row in rows:
        label = clean_line(row.get("label", ""))
        if not label:
            continue

        vals = []
        for date_idx, _ in enumerate(dates):
            val = row["values"].get(date_idx)
            vals.append(clean_line(val) if val else "-")

        out.append(f"{label}: " + " | ".join(vals))

    if notes:
        out.append("")
        out.append("Unassigned values / notes: " + " | ".join(clean_line(n) for n in notes if clean_line(n)))

    if review_needed:
        out.append("")
        out.append("TABLE_REVIEW_NEEDED")
        for reason in review_reasons:
            out.append(f"- {reason}")

    out.append("")
    return out

def format_parsed_table(title_lines, parsed_table):
    out = []

    seen_titles = set()
    for t in title_lines:
        tt = clean_line(t)
        if not tt:
            continue
        if tt not in seen_titles:
            out.append(tt)
            seen_titles.add(tt)

    if out:
        out.append("")

    rows = parsed_table.get("rows", [])
    notes = parsed_table.get("notes", [])
    review_needed = parsed_table.get("review_needed", False)
    review_reasons = parsed_table.get("review_reasons", [])

    for row in rows:
        label = clean_line(row.get("label", ""))
        if not label:
            continue

        values = row.get("values", {})
        if not values:
            continue

        max_col = max(values.keys()) if values else -1
        ordered_vals = []
        for col_idx in range(max_col + 1):
            val = values.get(col_idx)
            ordered_vals.append(clean_line(val) if val else "-")

        out.append(f"{label}: {' | '.join(ordered_vals)}")

    if notes:
        cleaned_notes = []
        seen_notes = set()
        for n in notes:
            nn = clean_line(n)
            if nn and nn not in seen_notes and not is_junk_footer_line(nn):
                seen_notes.add(nn)
                cleaned_notes.append(nn)

        if cleaned_notes:
            out.append("")
            out.append("Unassigned values / notes: " + " | ".join(cleaned_notes))

    if review_needed:
        out.append("")
        out.append("TABLE_REVIEW_NEEDED")
        for reason in review_reasons:
            out.append(f"- {clean_line(reason)}")

    return out
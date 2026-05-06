from pathlib import Path
import re


def clean_text(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_accents_basic(s: str) -> str:
    repl = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "ù": "u", "û": "u",
        "ï": "i", "î": "i",
        "ô": "o", "ö": "o",
        "ç": "c",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "À": "A", "Â": "A",
        "Ù": "U", "Û": "U",
        "Ï": "I", "Î": "I",
        "Ô": "O", "Ö": "O",
        "Ç": "C",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def norm(s: str) -> str:
    s = clean_text(s)
    s = s.replace("：", ":").replace("’", "'").replace("−", "-")
    s = s.replace("≥", ">=").replace("≤", "<=")
    s = strip_accents_basic(s).lower()
    return clean_text(s)


def compact_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9><=+/-]", "", norm(s))


def load_ocr_txt(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_page_block(lines: list[str], page_num: int) -> list[str]:
    start_pat = f"===== PAGE {page_num} ====="
    end_pat = f"===== PAGE {page_num + 1} ====="

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if clean_text(line) == start_pat:
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    for j in range(start_idx, len(lines)):
        if clean_text(lines[j]) == end_pat:
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(lines)

    page_lines = [clean_text(x) for x in lines[start_idx:end_idx]]
    return [x for x in page_lines if x]


SELECTED_CHARS = {"X", "x", "×", "•"}
UNSELECTED_CHARS = {"O", "o", "□", "C"}


def _marker_type(ch: str) -> str | None:
    if ch in SELECTED_CHARS:
        return "X"
    if ch in UNSELECTED_CHARS:
        return "O"
    return None


def is_lone_marker(line: str) -> str | None:
    s = clean_text(line)
    if len(s) == 1:
        return _marker_type(s)
    return None


def split_embedded_markers(line: str):
    line = clean_text(line)
    if not line:
        return []

    first = line[0]
    mt = _marker_type(first)
    if mt is not None:
        rest = clean_text(line[1:])
        rest_norm = rest.lower().replace(" ", "")
        if rest_norm == "ui":
            rest = "Oui"
        if rest:
            return [(mt, rest)]
        return [(mt, "")]

    m = re.match(r"^0\s+(.+)$", line)
    if m:
        return [("O", clean_text(m.group(1)))]

    parts = re.split(
        r'(?<=[a-z0-9éèêàùîïôç])\s*([XxOo×•])\s*(?=[A-ZÀ-Ü0-9><=])',
        line,
    )

    if len(parts) == 1:
        return [(None, line)]

    results = []
    if parts[0].strip():
        results.append((None, clean_text(parts[0])))

    i = 1
    while i + 1 < len(parts):
        raw_marker = parts[i]
        mt2 = _marker_type(raw_marker)
        marker = mt2 if mt2 else raw_marker.upper()
        text = clean_text(parts[i + 1])
        if text:
            results.append((marker, text))
        i += 2

    return results if results else [(None, line)]


def postprocess_lines(lines: list[str], replacements: list[tuple[str, str]] | None = None) -> list[str]:
    replacements = replacements or []

    replaced = []
    for line in lines:
        line = clean_text(line)
        for a, b in replacements:
            line = line.replace(a, b)
        replaced.append(clean_text(line))

    expanded = []
    for line in replaced:
        for pair in split_embedded_markers(line):
            expanded.append(pair)

    resolved = []
    i = 0
    while i < len(expanded):
        marker, text = expanded[i]
        if marker is not None and text == "":
            j = i + 1
            while j < len(expanded) and expanded[j][1] == "":
                j += 1
            if j < len(expanded) and expanded[j][0] is None:
                resolved.append((marker, expanded[j][1]))
                i = j + 1
                continue
        resolved.append((marker, text))
        i += 1

    cleaned = []
    prev = None
    for marker, text in resolved:
        text = clean_text(text)
        if not text:
            continue
        out_line = f"{marker} {text}" if marker else text
        if out_line == prev:
            continue
        cleaned.append(out_line)
        prev = out_line

    return cleaned


def parse_prefix_and_text(line: str):
    raw = clean_text(line)
    if not raw:
        return None, raw

    first = raw[0]
    mt = _marker_type(first)
    if mt is not None:
        rest = clean_text(raw[1:])
        rest_norm = rest.lower().replace(" ", "")
        if rest_norm == "ui":
            rest = "Oui"
        if rest:
            return mt, rest
        return mt, raw

    m = re.match(r"^0\s+(.+)$", raw)
    if m:
        return "O", clean_text(m.group(1))

    return None, raw


def line_select_state(line: str):
    prefix, rest = parse_prefix_and_text(line)
    if prefix == "X":
        return True, rest
    if prefix == "O":
        return False, rest
    return None, line


def text_matches_option(candidate_text: str, option_variant: str) -> bool:
    c = norm(candidate_text)
    o = norm(option_variant)
    if c == o:
        return True

    cc = compact_norm(candidate_text)
    oc = compact_norm(option_variant)
    if cc == oc:
        return True
    if len(oc) >= 3 and (cc.startswith(oc) or oc in cc):
        return True
    return False


def find_first_line_index(lines, anchors):
    for i, line in enumerate(lines):
        ln = norm(line)
        ln_compact = compact_norm(line)
        for a in anchors:
            an = norm(a)
            ac = compact_norm(a)
            if an in ln or (ac and ac in ln_compact):
                return i
    return None


def slice_section(lines, start_anchors, end_anchors=None, max_lines=20):
    start_idx = find_first_line_index(lines, start_anchors)
    if start_idx is None:
        return None

    end_idx = None
    if end_anchors:
        for j in range(start_idx + 1, min(len(lines), start_idx + max_lines + 1)):
            ln = lines[j]
            ln_norm = norm(ln)
            ln_compact = compact_norm(ln)
            for a in end_anchors:
                an = norm(a)
                ac = compact_norm(a)
                if an in ln_norm or (ac and ac in ln_compact):
                    end_idx = j
                    break
            if end_idx is not None:
                break

    if end_idx is None:
        end_idx = min(len(lines), start_idx + max_lines)

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "anchor_line": lines[start_idx],
        "lines": lines[start_idx:end_idx],
    }


def collect_line_evidence(section_lines):
    evidence = []
    for line in section_lines:
        state, content = line_select_state(line)
        prefix, _ = parse_prefix_and_text(line)
        evidence.append((prefix, content))
    return evidence


def parse_option_from_lines(section_lines, canonical_option, variants):
    matches = []
    line_evidence = collect_line_evidence(section_lines)

    for prefix, content in line_evidence:
        for v in variants:
            if text_matches_option(content, v):
                state = True if prefix == "X" else False if prefix == "O" else None
                matches.append({
                    "raw_line": content if prefix is None else f"{prefix} {content}",
                    "parsed_text": content,
                    "state": state,
                })
                break

    if not matches:
        return {
            "option": canonical_option,
            "found": False,
            "selected": False,
            "evidence": [],
        }

    x_hits = [m for m in matches if m["state"] is True]
    o_hits = [m for m in matches if m["state"] is False]

    return {
        "option": canonical_option,
        "found": True,
        "selected": bool(x_hits),
        "evidence": matches,
        "decision_source": (
            "ocr_prefix_X" if x_hits else
            "ocr_prefix_O" if o_hits else
            "plain_text_only"
        ),
    }


def postprocess_single_choice(option_results):
    x_selected = [
        x for x in option_results
        if x.get("found") and x.get("selected") and x.get("decision_source") == "ocr_prefix_X"
    ]

    if not x_selected:
        return option_results

    keep_names = {x["option"] for x in x_selected}
    out = []
    for item in option_results:
        item = dict(item)
        if item["option"] not in keep_names and item.get("selected"):
            item["selected"] = False
            item["override_reason"] = "single_choice_but_ocr_X_exists_elsewhere"
        out.append(item)
    return out


def apply_elimination_heuristic(option_results: list, single_choice: bool) -> list:
    if not single_choice:
        return option_results

    if any(x.get("selected") for x in option_results):
        return option_results

    found_options = [x for x in option_results if x.get("found")]
    if not found_options:
        return option_results

    explicit_o = [x for x in found_options if x.get("decision_source") == "ocr_prefix_O"]
    plain_only = [x for x in found_options if x.get("decision_source") == "plain_text_only"]
    explicit_x = [x for x in found_options if x.get("decision_source") == "ocr_prefix_X"]

    if explicit_x:
        return option_results

    candidate = None
    if len(plain_only) == 1 and len(explicit_o) == len(found_options) - 1:
        candidate = plain_only[0]
    elif len(found_options) == 1 and len(plain_only) == 1:
        candidate = plain_only[0]

    if candidate is None:
        return option_results

    out = []
    for item in option_results:
        item = dict(item)
        if item["option"] == candidate["option"]:
            item["selected"] = True
            item["decision_source"] = "elimination_heuristic"
        out.append(item)
    return out

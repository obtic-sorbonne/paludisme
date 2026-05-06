import re
from typing import List, Dict


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def resolve_overlaps(matches: List[Dict]) -> List[Dict]:
    """
    Keep highest-priority / longest match when overlaps occur.
    Lower priority value = stronger priority.
    """
    if not matches:
        return []

    matches = sorted(
        matches,
        key=lambda m: (m["start"], m.get("priority", 999), -(m["end"] - m["start"]))
    )

    kept = []
    for m in matches:
        conflict_idx = None
        for i, k in enumerate(kept):
            if overlaps(m["start"], m["end"], k["start"], k["end"]):
                conflict_idx = i
                break

        if conflict_idx is None:
            kept.append(m)
            continue

        k = kept[conflict_idx]
        m_len = m["end"] - m["start"]
        k_len = k["end"] - k["start"]
        m_pri = m.get("priority", 999)
        k_pri = k.get("priority", 999)

        replace = False
        if m_pri < k_pri:
            replace = True
        elif m_pri == k_pri and m_len > k_len:
            replace = True

        if replace:
            kept[conflict_idx] = m

    kept.sort(key=lambda x: x["start"])
    return kept


def apply_replacements(text: str, matches: List[Dict]):
    """
    Apply replacements from right to left so offsets stay stable.
    Returns (new_text, replacements_log)
    """
    if not matches:
        return text, []

    matches = resolve_overlaps(matches)
    out = text
    log = []

    for m in sorted(matches, key=lambda x: x["start"], reverse=True):
        original = out[m["start"]:m["end"]]
        out = out[:m["start"]] + m["replacement"] + out[m["end"]:]
        log.append({
            "original": original,
            "replacement": m["replacement"],
            "category": m["category"],
            "start": m["start"],
            "end": m["end"],
        })

    log.reverse()
    return out, log


def add_regex_matches(
    matches: List[Dict],
    text: str,
    pattern: str,
    replacement: str,
    category: str,
    priority: int,
    flags: int = 0,
    group: int = 1,
):
    for match in re.finditer(pattern, text, flags):
        start = match.start(group) if match.lastindex else match.start(0)
        end = match.end(group) if match.lastindex else match.end(0)
        matches.append({
            "start": start,
            "end": end,
            "replacement": replacement,
            "category": category,
            "priority": priority,
        })
from __future__ import annotations

import re


def clean_line(line: str) -> str:
    line = str(line).replace("\xa0", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def normalize_numeric_token(s: str) -> str:
    s = clean_line(s)
    if not s:
        return s
    s = s.replace("O", "0")
    if len(s) <= 3:
        s = s.replace("o", "0")
    s = s.replace("\u03bc", "\xb5")
    s = re.sub(r"\s+", " ", s)
    return s


def is_date_token(s: str) -> bool:
    s = clean_line(s).replace(".", "/")
    return bool(re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", s))


def normalize_date_token(s: str) -> str:
    s = clean_line(s).replace(".", "/")
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if not m:
        return s
    d, mth, y = m.groups()
    if len(y) == 2:
        y = "20" + y if int(y) < 30 else "19" + y
    return f"{int(d):02d}/{int(mth):02d}/{y}"


def looks_like_section_heading(s: str) -> bool:
    t = clean_line(s)
    if not t:
        return False

    measurement_unit_pat = re.compile(
        r"\(%\)|\(UI|\(/mm|\(g/|\(mmol|\(\xb5|\(\u03bc|\(10|\(pg|\(dl|\(l\)|\(ml\)",
        re.IGNORECASE,
    )

    if measurement_unit_pat.search(t):
        return False

    if t.endswith(":"):
        return True

    if len(t) <= 80 and re.fullmatch(r"[A-Za-zÀ-ÿ0-9'()/% .\-:&]+", t):
        alpha_count = sum(1 for c in t if c.isalpha())
        if alpha_count > 0:
            upper_ratio = sum(1 for c in t if c.isupper()) / alpha_count
            if upper_ratio > 0.55:
                return True

    return False
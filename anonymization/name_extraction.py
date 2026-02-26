"""
Patient name extraction from structured header fields in medical documents.

Parses names from patterns like:
  - "de l'enfant NDOUNKE DJATCHEU, MARVIN BRYAN"
  - "Nom patient  NDOUNKE DJATCHEU, MARVIN BI"
  - "Patient : NDOUNKE DJATCHEU, MARVIN BRYAN"
  - "IP0078733\nNDOUNKEDJATCHEU\nMARVIN BRYAN"

Generates all plausible variants (concatenated lastnames, truncated firstnames)
for thorough replacement.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Regex patterns targeting structured patient name fields.
# Input is clean text from iterate_items() (no markdown artifacts).
# Apostrophe may still be missing from OCR at the character level.
HEADER_PATTERNS = [
    # "de l'enfant LASTNAME, FIRSTNAME" (apostrophe may be missing from OCR)
    r"de\s+l[''']?enfant\s+([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    # "Nom patient  LASTNAME, FIRSTNAME"
    r"Nom\s+patient\s+([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    # "Patient : LASTNAME, FIRSTNAME" — same line
    r"Patient\s*:\s*([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    # "Patient" on one line, name on next (OCR line break)
    r"Patient\s*\n+\s*([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    # Urgences IP block: lastname on one line, firstname on next (same line only)
    r"IP\d+\s*\n\s*([A-ZÀ-Ü]{3,}(?:[A-ZÀ-Ü]*)?)\s*\n\s*([A-ZÀ-Ü][A-Za-zà-ÿ]+(?:[^\S\n]+[A-ZÀ-Ü][A-Za-zà-ÿ]+)*)",
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract(texts: list[str]) -> Optional[dict]:
    """
    Extract patient name from one or more document texts.

    Strategy: one subfolder = one patient. Different documents may have
    the name at different levels of completeness (lab results truncate
    firstnames, urgences concatenate lastnames). We scan ALL patterns
    across ALL documents, take the LONGEST match for firstnames, and
    prefer separated lastnames from other matches.

    Returns:
        {
            "lastnames":     ["NDOUNKE", "DJATCHEU"],
            "firstnames":    ["MARVIN", "BRYAN"],
            "all_tokens":    ["NDOUNKE", "DJATCHEU", "MARVIN", "BRYAN"],
            "full_variants": ["NDOUNKE DJATCHEU, MARVIN BRYAN", ...],
        }
        or None if no name found.
    """
    all_matches = []
    for pattern in HEADER_PATTERNS:
        for text in texts:
            match = re.search(pattern, text)
            if match:
                if match.lastindex and match.lastindex >= 2:
                    raw = f"{match.group(1)}, {match.group(2)}"
                else:
                    raw = match.group(1)
                all_matches.append(_normalize(raw))

    if not all_matches:
        return None

    # Parse all matches
    parsed = [_parse_name(m) for m in all_matches]

    # Take the longest match — most complete firstnames (BRYAN vs BI)
    best = max(parsed, key=lambda p: len(" ".join(p["firstnames"])))

    # If best has single concatenated lastname (NDOUNKEDJATCHEU), check if
    # another match has them separated (NDOUNKE DJATCHEU) — prefer separated.
    if len(best["lastnames"]) == 1:
        for other in parsed:
            if len(other["lastnames"]) > 1:
                # Verify it's the same name concatenated
                concat = "".join(other["lastnames"])
                if concat == best["lastnames"][0]:
                    best["lastnames"] = other["lastnames"]
                    break

    # Rebuild tokens and variants with merged info
    best["all_tokens"] = best["lastnames"] + best["firstnames"]
    best["full_variants"] = _build_variants(best["lastnames"], best["firstnames"])

    logger.info(f"Extracted patient name: {best['lastnames']} {best['firstnames']}")
    return best


def _parse_name(raw: str) -> dict:
    """Parse a raw name string into structured components."""
    if "," in raw:
        last_part, first_part = raw.split(",", 1)
    else:
        tokens = raw.split()
        last_part = " ".join(t for t in tokens if t == t.upper())
        first_part = " ".join(t for t in tokens if t != t.upper())

    lastnames = [t for t in last_part.split() if len(t.strip()) >= 2]
    firstnames = [t for t in first_part.split() if len(t.strip()) >= 2]
    full_variants = _build_variants(lastnames, firstnames)

    return {
        "lastnames": lastnames,
        "firstnames": firstnames,
        "all_tokens": lastnames + firstnames,
        "full_variants": full_variants,
    }


def _build_variants(lastnames: list[str], firstnames: list[str]) -> list[str]:
    """Generate all plausible name variant strings, sorted longest-first."""
    variants = set()
    ln_str = " ".join(lastnames)
    fn_str = " ".join(firstnames)

    if lastnames and firstnames:
        variants.add(f"{ln_str}, {fn_str}")
        variants.add(f"{ln_str} {fn_str}")

    # Concatenated lastnames (OCR artifact: NDOUNKEDJATCHEU)
    concat = ""
    if len(lastnames) > 1:
        concat = "".join(lastnames)
        variants.add(concat)
        if firstnames:
            variants.add(f"{concat}, {fn_str}")
            variants.add(f"{concat} {fn_str}")

    # Truncated firstnames (data-entry artifacts: MARVIN BRY, MARVI BRYAN, etc.)
    for i, fn in enumerate(firstnames):
        for length in range(2, len(fn)):
            trunc = fn[:length]
            partial = " ".join(firstnames[:i] + [trunc] + firstnames[i + 1 :])
            variants.add(f"{ln_str}, {partial}")
            variants.add(f"{ln_str} {partial}")
            if concat:
                variants.add(f"{concat}, {partial}")
                variants.add(f"{concat} {partial}")

    return sorted(variants, key=len, reverse=True)
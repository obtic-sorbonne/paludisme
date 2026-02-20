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
# Each captures the full name string (possibly with comma separator).
HEADER_PATTERNS = [
    r"de\s+l[''']enfant\s+([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    r"Nom\s+patient\s+([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    r"Patient\s*:\s*([A-ZÀ-Ü][A-ZÀ-Ü \-]+,\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
    # Urgences IP block: lastname on one line, firstname on next
    r"IP\d+\s*\n\s*([A-ZÀ-Ü]{3,}(?:[A-ZÀ-Ü]*)?)\s*\n\s*([A-ZÀ-Ü][A-Za-zà-ÿ]+(?:\s+[A-ZÀ-Ü][A-Za-zà-ÿ]+)*)",
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract(texts: list[str]) -> Optional[dict]:
    """
    Extract patient name from one or more document texts.

    Returns:
        {
            "lastnames":     ["NDOUNKE", "DJATCHEU"],
            "firstnames":    ["MARVIN", "BRYAN"],
            "all_tokens":    ["NDOUNKE", "DJATCHEU", "MARVIN", "BRYAN"],
            "full_variants": ["NDOUNKE DJATCHEU, MARVIN BRYAN", ...],
        }
        or None if no name found.
    """
    raw_matches = []
    for text in texts:
        for pattern in HEADER_PATTERNS:
            for match in re.finditer(pattern, text):
                if match.lastindex and match.lastindex >= 2:
                    raw = f"{match.group(1)}, {match.group(2)}"
                else:
                    raw = match.group(1)
                raw_matches.append(_normalize(raw))

    if not raw_matches:
        return None

    # Take the longest (most complete) match
    best = max(raw_matches, key=len)
    logger.info(f"Extracted patient name: '{best}'")

    # Split into lastnames / firstnames
    if "," in best:
        last_part, first_part = best.split(",", 1)
    else:
        tokens = best.split()
        last_part = " ".join(t for t in tokens if t == t.upper())
        first_part = " ".join(t for t in tokens if t != t.upper())

    lastnames = [t for t in last_part.split() if len(t.strip()) >= 2]
    firstnames = [t for t in first_part.split() if len(t.strip()) >= 2]
    all_tokens = lastnames + firstnames

    # Build matching variants
    full_variants = _build_variants(lastnames, firstnames)

    return {
        "lastnames": lastnames,
        "firstnames": firstnames,
        "all_tokens": all_tokens,
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
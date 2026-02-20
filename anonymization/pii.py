"""
PII detection: regex patterns for structured PII and person-name detectors.

Two categories:
  1. Structured PII (DOB, phone, email, address, identifiers) → replaced by tags
  2. Other people (doctors, staff, family) → replaced by [PERSONNE]
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Structured PII patterns
#    Each key maps to a list of regex patterns. The first capturing group (if
#    any) is what gets replaced; otherwise the full match is replaced.
# ---------------------------------------------------------------------------

STRUCTURED_PII = {
    "NIR": [
        r"\b([12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2})\b",
    ],
    "IDENTIFIANT": [
        r"NPI\s*:?\s*(\d{8,15})",
        r"(IP\d{5,10})",
        r"N[°o]?\s*de\s+séjour\s*:?\s*(\d{5,15})",
        r"NDA\s*:?\s*(\d{5,15})",
    ],
    "TELEPHONE": [
        r"\b(0[1-9][\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2})\b",
    ],
    "EMAIL": [
        r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
    ],
    "ADRESSE": [
        # Number + street type keyword + rest of line, optionally postal+city
        r"(\d{1,4}\s*(?:BIS|TER|B)?\s*(?:,\s*)?(?:rue|avenue|all[ée]e|boulevard|bd|impasse|passage|place|route|chemin|r[ée]sidence|cit[ée]|cours|square|lotissement)[^\n]{3,60}(?:\n\s*\d{5}\s+[A-ZÀ-Üa-zà-ÿ \-]+)?)",
        # "Patient adresse" field (lab results)
        r"[Pp]atient\s+adresse\s+(\d{1,4}[^\n]{3,60}\n\s*\d{5}\s+[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
        # "Adresse des parents :" block
        r"[Aa]dresse\s+des\s+parents\s*:\s*\n\s*(\d{1,4}[^\n]{3,60}\n\s*\d{5}\s+[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+)",
        # Standalone postal code (5 digits) + ALL-CAPS city
        r"^\s*(\d{5}\s+[A-ZÀ-Ü]{3,}(?:\s+[A-ZÀ-Ü]{2,})*)\s*$",
        # "Appartement à CITY"
        r"[Aa]ppartement\s+[àa]\s+([A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-]+?)(?=\s+avec|\s*$|\s*\n)",
    ],
}


def find_structured_pii(text: str) -> list[tuple[str, str, int, int]]:
    """
    Scan text for structured PII.
    Returns list of (pii_type, matched_text, start, end).
    """
    hits = []
    for pii_type, patterns in STRUCTURED_PII.items():
        for pattern in patterns:
            flags = re.MULTILINE if pattern.startswith("^") else 0
            for match in re.finditer(pattern, text, flags):
                captured = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)
                hits.append((pii_type, captured, start, end))
    return hits


# ---------------------------------------------------------------------------
# 2. Other-people detection (doctors, staff, family)
# ---------------------------------------------------------------------------

PERSON_PATTERNS = [
    # "Dr/Pr Firstname LASTNAME"
    r"(?:Dr|Pr|Professeur|Docteur)\.?\s+([A-ZÀ-Ü][a-zà-ÿ]+\s+[A-ZÀ-Ü][A-ZÀ-Ü\s]+?)(?=\s*[,;\n.\-]|\s+(?:PH|CCA|PU|MCU)|\s+\-|\s*$)",
    # "Dr LASTNAME"
    r"(?:Dr|Pr|Professeur|Docteur)\.?\s+([A-ZÀ-Ü][A-ZÀ-Ü]{2,})(?=\s*[,;\n.\-]|\s+(?:PH|CCA|PU|MCU)|\s+\-|\s*$)",
    # "Interne(s) : LASTNAME, Firstname"
    r"[Ii]nterne(?:s)?\s*(?:\(s\))?\s*:?\s*([A-ZÀ-Ü][A-ZÀ-Ü]+,\s*[A-ZÀ-Ü][a-zà-ÿ]+)",
    # "Copie à : LASTNAME, FIRSTNAME"
    r"[Cc]opie\s+[àa]\s*:\s*([A-ZÀ-Ü][A-ZÀ-Ü]+,\s*[A-ZÀ-Ü]+)",
    # "Validé par: Dr. LASTNAME, Firstname"
    r"[Vv]alid[ée]\s+par\s*:?\s*(?:Dr\.?\s+)?([A-ZÀ-Ü][A-ZÀ-Ü]+,?\s*[A-ZÀ-Ü]?[a-zà-ÿ]*)",
    # "CCA : Dr Firstname LASTNAME"
    r"CCA\s*:\s*(?:Dr\.?\s+)?([A-ZÀ-Ü][a-zà-ÿ]+\s+[A-ZÀ-Ü][A-ZÀ-Ü]+)",
    # "dr017215:DELGADO, David" (urgences timestamps)
    r"dr\d+\s*:\s*([A-ZÀ-Ü][A-ZÀ-Ü]+,\s*[A-ZÀ-Ü][a-zà-ÿ]+)",
    # "En accord avec Jean Yves"
    r"[Ee]n\s+accord\s+avec\s+([A-ZÀ-Ü][a-zà-ÿ]+(?:\s+[A-ZÀ-Ü][a-zà-ÿ]+)?)",
    # "Pr Firstname LASTNAME"
    r"\bPr\s+([A-ZÀ-Ü][a-zà-ÿ]+\s+[A-ZÀ-Ü][A-ZÀ-Ü]+)",
]

# Medical terms / place names that NER or regex might wrongly tag as persons
FALSE_POSITIVES = {
    "QUININE", "MALARONE", "DOLIPRANE", "NIVAQUINE", "FLAGYL", "CLAMOXYL",
    "ZYTHROMAX", "EMLA", "ADVIL", "PLASMODIUM", "GLASGOW",
    "ROBERT", "FONTAINE",
}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def find_other_people(
    text: str,
    patient_tokens: set[str],
    use_spacy: bool = True,
    nlp=None,
) -> list[str]:
    """
    Detect non-patient person names.
    Returns deduplicated list sorted longest-first.
    """
    found = []

    # Regex-based detection
    for pattern in PERSON_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE):
            name = _normalize(match.group(1))
            tokens_upper = {t.upper() for t in name.split()}
            if tokens_upper & patient_tokens:
                continue
            if tokens_upper & FALSE_POSITIVES:
                continue
            found.append(name)

    # spaCy NER detection (optional)
    if use_spacy and nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PER":
                name = _normalize(ent.text)
                tokens_upper = {t.upper() for t in name.split()}
                if tokens_upper & patient_tokens:
                    continue
                if tokens_upper & FALSE_POSITIVES:
                    continue
                if len(name) >= 3:
                    found.append(name)

    # Deduplicate, longest first
    seen = set()
    unique = []
    for name in found:
        key = name.upper()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    unique.sort(key=len, reverse=True)
    return unique
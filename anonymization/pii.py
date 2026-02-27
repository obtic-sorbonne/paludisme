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
        r"([I1l]P\d{5,10})",
        r"N[°o]?\s*de\s+séjour\s*:?\s*(\d{5,15})",
        r"NDA\s*:?\s*(\d{5,15})",
        # Barcode format: *DIGITS* or *DIGITS_DIGITS*
        r"\*(\d{10,15}[_\d]*)\*",
        # Standalone 10+ digit number NOT preceded by phone keyword (hospital ID)
        r"(?<!\d)(\d{10,15})(?!\d)",
    ],
    "TELEPHONE": [
        # Phone WITH separators (spaces/dots/hyphens) — always a phone
        r"\b(0[1-9][\s.\-]\d{2}[\s.\-]\d{2}[\s.\-]\d{2}[\s.\-]\d{2})\b",
        # Phone WITHOUT separators — only if preceded by phone context keyword
        # (avoids matching hospital barcodes/NPIs like 0502040015)
        r"(?:Tel|Tél|Fax|Télécopie|téléphone|téléphon)[^\n]{0,10}?(0[1-9]\d{8})\b",
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

    Processing order matters for disambiguation:
      1. NIR (social security — very specific format)
      2. TELEPHONE (formatted numbers or phone-context numbers)
      3. IDENTIFIANT (NPI, IP, barcode, standalone digit sequences)
      4. EMAIL
      5. ADRESSE

    Overlap detection: once a span is matched, later patterns skip it.
    This prevents e.g. "Tel.: 0148207966" being caught as both TELEPHONE
    and IDENTIFIANT.
    """
    hits = []
    matched_spans = []  # list of (start, end) tuples

    def _overlaps(start: int, end: int) -> bool:
        for s, e in matched_spans:
            if start < e and end > s:  # any overlap
                return True
        return False

    priority_order = ["NIR", "TELEPHONE", "IDENTIFIANT", "EMAIL", "ADRESSE"]
    for pii_type in priority_order:
        patterns = STRUCTURED_PII.get(pii_type, [])
        for pattern in patterns:
            flags = re.MULTILINE if pattern.startswith("^") else 0
            for match in re.finditer(pattern, text, flags):
                captured = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)
                if _overlaps(start, end):
                    continue
                hits.append((pii_type, captured, start, end))
                matched_spans.append((start, end))
    return hits


# ---------------------------------------------------------------------------
# 2. Other-people detection (doctors, staff, family)
#
# Strategy: high-precision regex anchored to role keywords (Dr, Pr, Interne,
# Copie à, Validé par, CCA). These contexts reliably indicate person names.
# spaCy NER is used only as a supplement with strict structural validation,
# because French NER on OCR'd medical text produces many false positives
# (drug names, lab terms, OCR garbage).
# ---------------------------------------------------------------------------

PERSON_PATTERNS = [
    # "Dr/Pr Firstname LASTNAME" — requires explicit title prefix
    r"(?:Dr|Pr|Professeur|Docteur)\.?\s+([A-ZÀ-Ü][a-zà-ÿ]+\s+(?:DE\s+(?:LOS\s+)?)?[A-ZÀ-Ü][A-ZÀ-Ü]{2,})(?=\s*[,;\n.\-]|\s+(?:PH|CCA|PU|MCU)|\s*$)",
    # "Dr LASTNAME" (no firstname) — must be 3+ uppercase letters
    r"(?:Dr|Pr|Professeur|Docteur)\.?\s+([A-ZÀ-Ü]{3,})(?=\s*[,;\n.\-]|\s+(?:PH|CCA|PU|MCU)|\s*$)",
    # "Interne(s) : LASTNAME, Firstname" — requires role prefix
    r"[Ii]nterne(?:s)?\s*(?:\(s\))?\s*:?\s*([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][a-zà-ÿ]+)",
    # "Copie à: LASTNAME, FIRSTNAME" — requires "Copie à" prefix
    r"[Cc]opie\s+[àa]\s*:?\s*([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ]+)",
    # "Validé par: Dr. LASTNAME, Firstname" — requires explicit name after "par"
    r"[Vv]alid[ée]\s+par\s*:?\s*(?:Dr\.?\s+)?([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][a-zà-ÿ]+)",
    # "CCA : Dr Firstname LASTNAME"
    r"CCA\s*:\s*(?:Dr\.?\s+)?([A-ZÀ-Ü][a-zà-ÿ]+\s+[A-ZÀ-Ü]{3,})",
    # "dr017215:DELGADO, David" (urgences timestamps)
    r"dr\d+\s*:\s*([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][a-zà-ÿ]+)",
    # "En accord avec Firstname Lastname"
    r"[Ee]n\s+accord\s+avec\s+([A-ZÀ-Ü][a-zà-ÿ]{2,}(?:\s+[A-ZÀ-Ü][a-zà-ÿ]{2,})?)",
    # "Par.....: LASTNAME, Firstname" (lab result technician/validator)
    r"Par\.{2,}\s*:\s*([A-ZÀ-Ü][A-ZÀ-Ü-]+,\s*[A-ZÀ-Ü][a-zà-ÿ]+)(?:\s*,)?",
    # "Médecin(s) senior(s) : LASTNAME, Firstname" or "LASTNAME Firstname"
    r"[Mm][ée]decin(?:s)?\s*(?:\(s\))?\s*senior(?:s)?\s*(?:\(s\))?\s*:\s*([A-ZÀ-Ü]{3,},?\s*[A-ZÀ-Ü][a-zà-ÿ]+(?:\s+[A-ZÀ-Ü][a-zà-ÿ]+)?)",
    # "SIRIEZ, JEAN YVES" or "LASTNAME, FIRSTNAME FIRSTNAME" in MAITRE line
    r"MAITRE=\w+\s*\n\s*([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü]{3,}(?:\s+[A-ZÀ-Ü]{3,})?)",
]

# Known false positives — drug names and terms that structurally resemble names.
# Keep this minimal: structural validation handles most cases.
FALSE_POSITIVES = {
    # Drugs (various OCR spellings)
    "QUININE", "MALARONE", "DOLIPRANE", "NIVAQUINE", "FLAGYL", "CLAMOXYL",
    "ZYTHROMAX", "ZITHROMAX", "EMLA", "ADVIL", "CLAMOXYLET",
    # Medical/scientific terms that look like names
    "PLASMODIUM", "GLASGOW", "SHIGELLA",
    # Hospital name parts (also valid lastnames — but in this context, not people)
    "ROBERT", "FONTAINE", "DEBRÉ", "DEBRE",
    # Medical department terms that spaCy/regex tag as person names
    "DIABÉTOLOGIE", "DIABETOLOGIE", "PÉDIATRIQUE", "PEDIATRIQUE",
    "ENDOCRINOLOGIE", "HEMATOLOGIE", "HÉMATOLOGIE",
}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _is_plausible_name(name: str) -> bool:
    """
    Structural validation: does this string look like a person's name?

    Filters out OCR garbage, medical terms, abbreviations, and common
    French words that NER or loose regex might tag as person names.
    """
    # Must contain only letters, spaces, hyphens, commas, apostrophes
    if re.search(r"[0-9_.@/\\:;(){}]", name):
        return False

    # Reject strings containing dots (lab values like "Conc.corp.moy")
    if "." in name:
        return False

    tokens = [t for t in name.replace(",", " ").split() if t]

    if not tokens:
        return False

    # Every token must be at least 2 characters
    if any(len(t) < 2 for t in tokens):
        return False

    # Need at least one token of 4+ letters
    if not any(len(t) >= 4 for t in tokens):
        return False

    # Single-word: must be all-uppercase and 4+ chars (e.g. "SIRIEZ" from "Dr SIRIEZ").
    # Reject single titlecase words — they're almost always medical terms
    # (Cutané, Lipemie, Zithromax, Clamoxylet, Tenfant, Lenfant, Urbain...).
    if len(tokens) == 1:
        word = tokens[0]
        if not word.isupper():
            return False
        if len(word) < 4:
            return False

    # Multi-word: reject if any token is all-lowercase (catches "Shigella flexneri",
    # "cellulaires sur lame" etc. — real names always start with uppercase)
    if len(tokens) > 1:
        for t in tokens:
            if t.islower():
                return False

    return True


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

    # --- Regex-based detection (high precision, anchored to role keywords) ---
    for pattern in PERSON_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE):
            name = _normalize(match.group(1))
            tokens_upper = {t.upper() for t in name.replace(",", " ").split()}
            if tokens_upper & patient_tokens:
                continue
            if tokens_upper & FALSE_POSITIVES:
                continue
            if not _is_plausible_name(name):
                continue
            found.append(name)

    # --- spaCy NER (supplement only, strict filtering) ---
    # French NER on OCR'd medical text is very noisy: drug names, bacteria,
    # lab terms, OCR artifacts all get tagged as PER. We only accept spaCy
    # matches that look structurally like "Firstname LASTNAME" or "LASTNAME".
    if use_spacy and nlp is not None:
        # Name must match: "Firstname LASTNAME" or "LASTNAME, Firstname"
        # with minimum 3 chars per word (filters "Hep Li", "Réa Debré" etc.)
        name_structure = re.compile(
            r"^[A-ZÀ-Ü][a-zà-ÿ]{2,}\s+(?:DE\s+(?:LOS\s+)?)?[A-ZÀ-Ü]{3,}$"   # Firstname LASTNAME
            r"|^[A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][a-zà-ÿ]{2,}$"                        # LASTNAME, Firstname
            r"|^[A-ZÀ-Ü][a-zà-ÿ]{2,}\s+[A-ZÀ-Ü][a-zà-ÿ]{2,}$"                # Firstname Lastname
        )
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ != "PER":
                continue
            name = _normalize(ent.text)
            if not name_structure.match(name):
                continue
            tokens_upper = {t.upper() for t in name.replace(",", " ").split()}
            if tokens_upper & patient_tokens:
                continue
            if tokens_upper & FALSE_POSITIVES:
                continue
            if not _is_plausible_name(name):
                continue
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
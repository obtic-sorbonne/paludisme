import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

IGNORE_TOKENS = {
    "NOM", "PRÉNOM", "PRENOM", "ETHNICITÉ", "ETHNICITE", "SEXE", "AGE",
    "NSP", "OUI", "NON", "DATE", "NAISSANCE", "LOCALISATION", "PAYS",
    "RÉSIDENCE", "RESIDENCE", "DURÉE", "DUREE", "CONSULTATION",
    "DIAGNOSTIC", "BIOLOGIQUE", "PATIENT", "AFRICAIN", "CAUCASIEN",
    "ASIATIQUE", "AUTRE", "FRANCE", "METROPOLITAINE", "ACCUEIL",
    "DÉCONNECTER", "DECONNECTER", "ANNÉE", "ANNEE", "ID", "FICHE",
    "HOPITAL", "ROBERT", "DEBRE", "SERVICE", "URGENCES", "PARIS",
    "HEURE", "SIGNATURE",
}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _clean_token(tok: str) -> str:
    return tok.strip(" ,;:.|/-_[](){}*•")


def _is_valid_name_token(tok: str) -> bool:
    tok = _clean_token(tok)
    if len(tok) < 1:
        return False
    if any(ch.isdigit() for ch in tok):
        return False
    if tok.upper() in IGNORE_TOKENS:
        return False
    return True


def _build_variants(lastnames: list[str], firstnames: list[str]) -> list[str]:
    variants = set()

    ln = " ".join(lastnames).strip()
    fn = " ".join(firstnames).strip()

    if ln and fn:
        variants.add(f"{ln}, {fn}")
        variants.add(f"{ln} {fn}")

    if fn:
        variants.add(fn)

    if len(lastnames) > 1:
        concat = "".join(lastnames)
        variants.add(concat)
        if fn:
            variants.add(f"{concat}, {fn}")
            variants.add(f"{concat} {fn}")

    return sorted({v for v in variants if v}, key=len, reverse=True)


def _parse_name(last_raw: str, first_raw: str) -> Optional[dict]:
    last_tokens = [_clean_token(t) for t in last_raw.split() if _is_valid_name_token(t)]
    first_tokens = [_clean_token(t) for t in first_raw.split() if _is_valid_name_token(t)]

    if not last_tokens and not first_tokens:
        return None

    return {
        "lastnames": last_tokens,
        "firstnames": first_tokens,
        "all_tokens": last_tokens + first_tokens,
        "full_variants": _build_variants(last_tokens, first_tokens),
    }


def extract(texts: list[str]) -> Optional[dict]:
    candidates = []

    for text in texts:
        # 1) Standard raw OCR / merged final-output label style
        # Supports:
        #   Nom: Boongo
        #   [3] Nom: Boongo
        #   [1] Prénom: Mereline
        nom_match = re.search(
            r"(?mi)^\s*(?:\[\d+\]\s*)?Nom\s*:\s*([^\n]{1,80})\s*$",
            text,
        )
        prenom_match = re.search(
            r"(?mi)^\s*(?:\[\d+\]\s*)?Pr[ée]nom\s*:\s*([^\n]{1,80})\s*$",
            text,
        )

        if nom_match and prenom_match:
            parsed = _parse_name(nom_match.group(1), prenom_match.group(1))
            if parsed and parsed["lastnames"] and parsed["firstnames"]:
                candidates.append(parsed)

        # 2) CRU / urgences style
        m = re.search(
            r"(?mi)^\s*([A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,})*)\s*\n\s*N[ée]\s*\(?.*?\)?\s*:\s*\n\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ]*(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ]*)*)\s*\n\s*Naiss",
            text,
        )
        if m:
            parsed = _parse_name(m.group(1), m.group(2))
            if parsed:
                candidates.append(parsed)

        # 3) Lab block
        m = re.search(
            r"(?mi)Nom\s+patient\s*\n\s*([A-ZÀ-Ü]{2,}(?:\s+[A-ZÀ-Ü]{2,})*)\s*\n\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ]*(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ]*)*)\s*\n\s*Date\s+naissance",
            text,
        )
        if m:
            parsed = _parse_name(m.group(1), m.group(2))
            if parsed:
                candidates.append(parsed)

    candidates = [c for c in candidates if c["lastnames"] and c["firstnames"]]
    if not candidates:
        return None

    def score(c):
        return (
            len(c["firstnames"]),
            len(c["lastnames"]),
            sum(len(x) for x in c["firstnames"]),
            sum(len(x) for x in c["lastnames"]),
        )

    best = max(candidates, key=score)
    best["all_tokens"] = best["lastnames"] + best["firstnames"]
    best["full_variants"] = _build_variants(best["lastnames"], best["firstnames"])

    logger.info(
        "Extracted patient: %s %s",
        " ".join(best["lastnames"]),
        " ".join(best["firstnames"]),
    )
    return best
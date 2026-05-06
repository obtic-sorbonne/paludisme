import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STRUCTURED_PII = {
    "NIR": [
        r"\b([12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2})\b",
    ],
    "DOB": [
        r"(?:Date\s+de\s+naissance|N[ée]e?\s+le|N[ée]\(e\)|Naiss)\s*[: ]+\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    ],
    "IDENTIFIANT": [
        r"\bNIP\s*[: ]+\s*([A-Za-z0-9_-]{6,20})",
        r"\bID\s+patient\s*[: ]+\s*([A-Za-z0-9_-]{2,30})",
        r"(?mi)^\s*Id\s*[: ]+\s*([A-Za-z0-9_-]*\d[A-Za-z0-9_-]{2,20})\s*$",
        r"\bID\s+Correspondant\s*[: ]+\s*([A-Za-z0-9_-]{2,30})",
        r"\bNPI\s*[: ]+\s*(\d{6,15})",
        r"\bN[°o]?\s*de\s+séjour\s*[: ]+\s*([A-Za-z0-9_-]{2,30})",
        r"\bNDA\s*[: ]+\s*(\d{5,15})",
        r"\b([I1l]P\d{5,12})\b",
        r"\*(\d{8,20}[_\d]*)\*",
    ],
    "TELEPHONE": [
        r"\b(0[1-9][\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2})\b",
        r"(?:Tel|T[ée]l|T[ée]l[ée]copie|Fax|Portable)\s*[: ]+\s*(0[1-9][\d\s.\-]{8,14})",
    ],
    "EMAIL": [
        r"\b([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+)\b",
    ],
    "ADRESSE": [
        r"(\d{1,4}\s*(?:bis|ter|BIS|TER)?\s*(?:rue|avenue|av\.?|boulevard|bd|impasse|passage|place|route|chemin|all[ée]e|cours|square)\s+[^\n]{3,120})",
        r"(\b\d{5}\s+[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ0-9 \-]{2,}\b)",
        r"(?mis)(?:Adresse\s+des\s+parents|Patient\s+adresse)\s*:\s*\n\s*([^\n]{3,120}\n\s*\d{5}\s+[A-ZÀ-ÜA-Za-zÀ-ÿ0-9 \-]{2,80})",
        r"(?mis)(\d{1,4}\s*(?:bis|ter|BIS|TER)?\s*(?:rue|avenue|av\.?|boulevard|bd|impasse|passage|place|route|chemin|all[ée]e|cours|square)\s+[^\n]{3,120}\n\s*\d{5}\s+[A-ZÀ-ÜA-Za-zÀ-ÿ0-9 \-]{2,80})",
        r"(\b\d{1,4}(?:bis|ter)?(?:RUE|AVENUE|AV\.?|BOULEVARD|BD|IMPASSE|PASSAGE|PLACE|ROUTE|CHEMIN|ALLEE|COURS|SQUARE)[A-ZÀ-Üa-zà-ÿ0-9\- ]{2,80}\d{5}[A-ZÀ-Üa-zà-ÿ0-9\- ]{2,40}\b)",
        r"(\(\s*\d{5}\s*\)\s*[A-ZÀ-ÜA-Za-zÀ-ÿ0-9 \-]{2,40})",
        r"(\b\d{5}\s*[)\]）]?\s*[A-ZÀ-ÜA-Za-zÀ-ÿ0-9 \-]{2,40}\b)",
        r"(\b\d{5}[A-ZÀ-ÜA-Za-zÀ-ÿ0-9\-]{2,40}\b)",
    ],
}

HOSPITAL_PATTERNS = [
    r"((?:H[ôo]pital|Hopital|ASSISTANCE\s+PUBLIQUE\s+HOPITAUX\s+DE\s+PARIS|AP-HP)\s+[A-ZÀ-ÜA-Za-zÀ-ÿ \-']*)",
    r"((?:Service|Unit[ée]|Laboratoire|Urgences)\s+(?:de|des|du)?\s*[A-ZÀ-ÜA-Za-zÀ-ÿ \-']{2,})",
]

PERSON_PATTERNS = [
    r"(?:Dr|Docteur|Pr|Professeur)\.?\s+([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,3})",

    r"(?:Valid[ée]\s+par|Relecture\s+par|Copie\s+[àa]|M[ée]decin(?:\s+traitant)?|M[ée]decin\(s\)\s+senior\(s\)|Interne\(s\)|Etudiant\s+hospitalier)\s*[: ]+\s*(?:M\.?\s+)?([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:,\s*[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+)?(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){0,3})",

    r"\b([A-ZÀ-Ü]{3,},\s*[A-ZÀ-Ü][a-zà-ÿ'\-]{2,})\b",

    r"\b(Monsieur\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:,\s*[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+)?)\b",

    r"(?:M[ée]decin\(s\)\s+senior\(s\)|Interne\(s\)|Relecture\s+par)\s*[: ]+\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,4})",
]

FAMILY_CONTEXT_PATTERNS = [
    r"[Pp]ersonne\s+[àa]\s+pr[ée]venir\s*:\s*([^\n]+)",
    r"[Pp]ERSONNE\s+[A-ZÀ-Ü ]*PREVENIR\s*:\s*([^\n]+)",
    r"(?:M[èe]re|P[èe]re|Parent|Parents|Accompagnant\(s\)|Personne\s+[àa]\s+pr[ée]venir)\s*:\s*([^\n]+)",
    r"[Aa]ccompagn[ée]?\s+(?:par|de)\s*:\s*([^\n]+)",
]

NON_PERSON_ROLE_LABELS = {
    "TECHNICIEN", "HEMATO", "JOUR", "NUIT", "MEDECIN", "SENIOR",
    "INTERNE", "ETUDIANT", "HOSPITALIER", "VALIDE", "VALIDEE",
    "COPIE", "RELECTURE", "SERVICE", "URGENCES", "LABORATOIRE",
    "PEDIATRIE", "PEDIATRIQUES", "PEDIATRIQUE", "URGENT",
    "CION", "CDERGENT", "CDARGENT",
}

FALSE_POSITIVES = {
    "PLASMODIUM", "FALCIPARUM", "MALARONE", "QUININE",
    "URGENCES", "LABORATOIRE", "SERVICE", "PARIS", "BOBIGNY",
    "PEDIATRIE", "PEDIATRIQUES", "PEDIATRIQUE", "URGENT",
    "CION", "CDERGENT", "CDARGENT",
}

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _is_plausible_person(name: str) -> bool:
    if re.search(r"[0-9@_/\\:;(){}]", name):
        return False
    toks = [t for t in name.replace(",", " ").split() if t]
    if not toks or len(toks) < 2:
        return False
    if any(t.upper() in FALSE_POSITIVES for t in toks):
        return False
    if all(t.upper() in NON_PERSON_ROLE_LABELS for t in toks):
        return False
    for t in toks:
        clean = t.strip("-,.' ")
        if len(clean) < 1:
            return False
        if not re.match(r"^[A-Za-zÀ-ÿ'\-]+$", clean):
            return False
    return True

def find_structured_pii(text: str) -> list[tuple[str, str, int, int]]:
    hits = []
    spans = []

    def overlaps(start, end):
        for s, e in spans:
            if start < e and end > s:
                return True
        return False

    for pii_type in ["NIR", "DOB", "TELEPHONE", "IDENTIFIANT", "EMAIL", "ADRESSE"]:
        for pattern in STRUCTURED_PII.get(pii_type, []):
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)
                captured = match.group(1) if match.lastindex else match.group(0)
                if overlaps(start, end):
                    continue
                hits.append((pii_type, captured, start, end))
                spans.append((start, end))
    return hits


def _split_family_contact_block(raw: str) -> list[str]:
    text = _normalize(raw)

    text = re.sub(r"(?i)\bP[èe]re\b", " ", text)
    text = re.sub(r"(?i)\bM[èe]re\b", " ", text)
    text = re.sub(r"(?i)\bParent\b", " ", text)
    text = re.sub(r"(?i)\bParents\b", " ", text)
    text = re.sub(r"(?i)\bP[èe]re\s*-\s*M[èe]re\b", " ", text)

    text = text.replace(";", ",")
    text = _normalize(text)

    parts = [p.strip(" ,") for p in text.split(",") if p.strip(" ,")]

    names = []
    for part in parts:
        part = _normalize(part)
        part = re.sub(r"^(?:-|:|\s)+", "", part).strip()

        if _is_plausible_person(part):
            names.append(part)

    return names


def find_family_contacts(text: str, patient_tokens: set[str]) -> list[str]:
    found = []

    for pattern in FAMILY_CONTEXT_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE):
            raw = match.group(1)
            for name in _split_family_contact_block(raw):
                if _is_plausible_person(name):
                    found.append(name)

    return sorted(set(found), key=len, reverse=True)


def find_other_people(
    text: str,
    patient_tokens: set[str],
    use_spacy: bool = False,
    nlp: Optional[object] = None,
) -> list[str]:
    found = []
    for pattern in PERSON_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            name = _normalize(match.group(1))
            toks = {t.upper() for t in name.replace(",", " ").split()}
            if toks and toks.issubset(patient_tokens):
                continue
            if _is_plausible_person(name):
                found.append(name)

    if use_spacy and nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ != "PER":
                continue
            name = _normalize(ent.text)
            toks = {t.upper() for t in name.replace(",", " ").split()}
            if toks and toks.issubset(patient_tokens):
                continue
            if _is_plausible_person(name):
                found.append(name)

    return sorted(set(found), key=len, reverse=True)


def find_hospitals_and_services(text: str) -> list[tuple[str, str]]:
    found = []
    for pattern in HOSPITAL_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            entity = _normalize(match.group(1))
            if not entity:
                continue
            upper = entity.upper()
            if upper.startswith("SERVICE") or upper.startswith("URGENCES") or upper.startswith("LABORATOIRE"):
                found.append(("service", entity))
            else:
                found.append(("hospital", entity))

    seen = set()
    unique = []
    for cat, ent in found:
        key = (cat, ent.upper())
        if key not in seen:
            seen.add(key)
            unique.append((cat, ent))
    unique.sort(key=lambda x: len(x[1]), reverse=True)
    return unique
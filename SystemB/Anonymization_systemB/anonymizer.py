import re
import logging
import unicodedata
from collections import defaultdict

from . import pii, pseudonyms
from .replacement_utils import apply_replacements

logger = logging.getLogger(__name__)


def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_variants(name: str) -> list[str]:
    """
    Build useful variants for a detected full name.
    Example:
      'MERCIER Jean Christophe'
      -> ['MERCIER Jean Christophe', 'Jean Christophe MERCIER']
    """
    name = re.sub(r"\s+", " ", name).strip(" ,;-")
    parts = name.split()
    if len(parts) < 2:
        return [name]

    variants = {name}

    surname = parts[0]
    given = " ".join(parts[1:])
    variants.add(f"{given} {surname}".strip())

    return sorted({v for v in variants if v}, key=len, reverse=True)


def _add_global_entity_matches(matches, text, entities, replacement, category, priority):
    """
    Once an entity is known, anonymize all occurrences of it across the document.
    """
    seen = set()

    for ent in entities:
        if not ent or len(ent.strip()) < 2:
            continue

        for variant in _name_variants(ent):
            key = _norm_key(variant)
            if key in seen:
                continue
            seen.add(key)

            pattern = re.escape(variant)
            for m in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    "start": m.start(),
                    "end": m.end(),
                    "replacement": replacement,
                    "category": category,
                    "priority": priority,
                })


def anonymize(text: str, patient_info: dict, patient_id: str = "001", nlp=None):
    result = text
    stats = defaultdict(int)
    matches = []

    patient_tokens_upper = {t.upper() for t in patient_info.get("all_tokens", [])}
    patient_map = pseudonyms.generate_patient_map(patient_info, patient_id=patient_id)
    patient_tag = f"[PATIENT_{patient_id}]"

    # -------------------------
    # 1) Detect entities first
    # -------------------------
    family_names = pii.find_family_contacts(result, patient_tokens_upper)

    other_people = pii.find_other_people(
        result,
        patient_tokens_upper,
        use_spacy=(nlp is not None),
        nlp=nlp,
    )

    hospital_entities = pii.find_hospitals_and_services(result)
    structured_hits = pii.find_structured_pii(result)

    hospitals = [ent for cat, ent in hospital_entities if cat == "hospital"]
    services = [ent for cat, ent in hospital_entities if cat == "service"]

    # Extra labeled staff detection
    labeled_staff = []
    label_patterns = [
        r"(?im)Médecin\(s\)\s+senior\(s\)\s*:\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,3})",
        r"(?im)Interne\(s\)\s*:\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,3})",
        r"(?im)Relecture\s+par\s*:\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,3})",
        r"(?im)Conclusion médicale UHCD\s*\(\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+){1,3})",
    ]
    for pat in label_patterns:
        for m in re.finditer(pat, result):
            labeled_staff.append(m.group(1).strip())

    all_staff = sorted(set(other_people + labeled_staff), key=len, reverse=True)

    # -------------------------
    # 2) Family/contact
    # -------------------------
    _add_global_entity_matches(
        matches, result, family_names, "[ANONYMIZED]", "family_contact", 1
    )

    # -------------------------
    # 3) Patient full variants
    # -------------------------
    for variant in patient_info.get("full_variants", []):
        for m in re.finditer(re.escape(variant), result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": patient_tag,
                "category": "patient_name",
                "priority": 2,
            })

    # -------------------------
    # 4) Patient single tokens
    # -------------------------
    for real_token, pseudo_token in patient_map.items():
        if len(real_token) < 2:
            continue
        if " " in real_token.strip():
            pattern = re.escape(real_token)
        else:
            pattern = r"\b" + re.escape(real_token) + r"\b"
        for m in re.finditer(pattern, result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": pseudo_token,
                "category": "patient_name",
                "priority": 3,
            })

    # -------------------------
    # 5) Hospitals / services
    # -------------------------
    _add_global_entity_matches(
        matches, result, hospitals, "[ANONYMIZED]", "hospital", 4
    )
    _add_global_entity_matches(
        matches, result, services, "[ANONYMIZED]", "service", 4
    )

    # -------------------------
    # 6) Structured PII
    # -------------------------
    for pii_type, _captured, start, end in structured_hits:
        if pii_type == "DOB":
            continue
        matches.append({
            "start": start,
            "end": end,
            "replacement": "[ANONYMIZED]",
            "category": pii_type.lower(),
            "priority": 5,
        })

    # Also anonymize the address label itself
    for m in re.finditer(r"(?im)^\s*Adresse\s+des\s+parents\s*:\s*$", result):
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "replacement": "[ANONYMIZED]",
            "category": "adresse",
            "priority": 5,
        })

    # -------------------------
    # 7) Staff / other people
    # -------------------------
    _add_global_entity_matches(
        matches, result, all_staff, "[ANONYMIZED]", "other_person", 6
    )

    anonymized_text, replacements = apply_replacements(result, matches)

    for r in replacements:
        stats[r["category"]] += 1

    return anonymized_text, dict(stats), replacements
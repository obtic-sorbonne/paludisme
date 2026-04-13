import re
import logging
from collections import defaultdict

from . import pii, pseudonyms
from .replacement_utils import apply_replacements

logger = logging.getLogger(__name__)


def anonymize(text: str, patient_info: dict, patient_id: str = "001", nlp=None):
    result = text
    stats = defaultdict(int)
    matches = []

    patient_tokens_upper = {t.upper() for t in patient_info.get("all_tokens", [])}
    patient_map = pseudonyms.generate_patient_map(patient_info, patient_id=patient_id)
    patient_tag = f"[PATIENT_{patient_id}]"

    # Family/contact first
    family_names = pii.find_family_contacts(result, patient_tokens_upper)
    family_map = pseudonyms.assign_entity_tags(family_names, "CONTACT")
    for name in family_map:
        for m in re.finditer(re.escape(name), result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": "[ANONYMIZED]",
                "category": "family_contact",
                "priority": 1,
            })

    # Full patient variants
    for variant in patient_info.get("full_variants", []):
        for m in re.finditer(re.escape(variant), result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": patient_tag,
                "category": "patient_name",
                "priority": 2,
            })

    # Patient single tokens
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

    # Hospital/service anonymization
    for cat, ent in pii.find_hospitals_and_services(result):
        for m in re.finditer(re.escape(ent), result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": "[ANONYMIZED]",
                "category": cat,
                "priority": 4,
            })

    # Structured PII, but keep DOB
    for pii_type, _captured, start, end in pii.find_structured_pii(result):
        if pii_type == "DOB":
            continue
        matches.append({
            "start": start,
            "end": end,
            "replacement": "[ANONYMIZED]",
            "category": pii_type.lower(),
            "priority": 5,
        })

    # Staff / other people
    other_people = pii.find_other_people(
        result,
        patient_tokens_upper,
        use_spacy=(nlp is not None),
        nlp=nlp,
    )
    other_map = pseudonyms.assign_entity_tags(other_people, "STAFF")
    for name in other_map:
        for m in re.finditer(re.escape(name), result, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "replacement": "[ANONYMIZED]",
                "category": "other_person",
                "priority": 6,
            })

    anonymized_text, replacements = apply_replacements(result, matches)

    for r in replacements:
        stats[r["category"]] += 1

    return anonymized_text, dict(stats), replacements
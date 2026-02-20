"""
Core anonymization engine.

Applies 4 layers in order:
  A) Multi-word patient name variants (longest first → avoids partial matches)
  B) Individual patient name tokens (with word boundaries)
  C) Structured PII (DOB, phone, email, address, identifiers)
  D) Other people's names (regex + optional spaCy NER)
"""

import re
import logging
from collections import defaultdict

from . import pii

logger = logging.getLogger(__name__)


def anonymize(
    text: str,
    patient_info: dict,
    pseudonym_map: dict[str, str],
    nlp=None,
) -> tuple[str, dict[str, int], list[dict]]:
    """
    Anonymize a single document text.

    Args:
        text:          raw document text
        patient_info:  from name_extraction.extract()
        pseudonym_map: from pseudonyms.generate()
        nlp:           spaCy Language model (or None)

    Returns:
        (anonymized_text, stats_dict, replacements_log)
        replacements_log: list of {"original": ..., "replacement": ..., "category": ...}
    """
    result = text
    stats = defaultdict(int)
    replacements = []
    patient_tokens_upper = {t.upper() for t in patient_info["all_tokens"]}

    # --- Layer A: Multi-word patient name variants (longest first) ---
    for variant in patient_info["full_variants"]:
        pseudo_variant = variant
        for real, pseudo in pseudonym_map.items():
            pseudo_variant = pseudo_variant.replace(real, pseudo)

        pattern = re.escape(variant)
        matches = re.findall(pattern, result, re.IGNORECASE)
        if matches:
            result = re.sub(pattern, pseudo_variant, result, flags=re.IGNORECASE)
            stats["patient_name_multiword"] += len(matches)
            for m in matches:
                replacements.append({"original": m, "replacement": pseudo_variant, "category": "patient_name"})

    # --- Layer B: Individual patient name tokens (word boundaries) ---
    for real_token, pseudo_token in pseudonym_map.items():
        if len(real_token) < 2:
            continue
        pattern = r"\b" + re.escape(real_token) + r"\b"
        matches = re.findall(pattern, result, re.IGNORECASE)
        if matches:
            result = re.sub(pattern, pseudo_token, result, flags=re.IGNORECASE)
            stats["patient_name_token"] += len(matches)
            for m in matches:
                replacements.append({"original": m, "replacement": pseudo_token, "category": "patient_name"})

    # --- Layer C: Structured PII ---
    for pii_type, captured, _, _ in pii.find_structured_pii(result):
        result = result.replace(captured, "[ANONYMIZED]", 1)
        stats[pii_type] += 1
        replacements.append({"original": captured, "replacement": "[ANONYMIZED]", "category": pii_type})

    # --- Layer D: Other people ---
    other_names = pii.find_other_people(
        result,
        patient_tokens_upper,
        use_spacy=(nlp is not None),
        nlp=nlp,
    )
    for name in other_names:
        pattern = re.escape(name)
        count = len(re.findall(pattern, result, re.IGNORECASE))
        if count:
            result = re.sub(pattern, "[ANONYMIZED]", result, flags=re.IGNORECASE)
            stats["other_person"] += count
            replacements.append({"original": name, "replacement": "[ANONYMIZED]", "category": "other_person"})

    return result, dict(stats), replacements
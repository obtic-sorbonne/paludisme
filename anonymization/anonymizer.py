"""
Core anonymization engine.

Applies 5 layers in order:
  A)  Multi-word patient name variants (longest first → avoids partial matches)
  A2) Family/contact names from "Personne à prévenir", "accompagné", etc.
      These are ALWAYS PII (parents, guardians). Detected from context keywords,
      then replaced GLOBALLY so they're caught everywhere in the dossier.
  B)  Individual patient name tokens (with word boundaries)
  C)  Structured PII (DOB, phone, email, address, identifiers)
  D)  Other people's names (regex + optional spaCy NER)
"""

import re
import logging
from collections import defaultdict

from . import pii

logger = logging.getLogger(__name__)

# Contexts where names are family members / contacts — always PII.
# Capture group = the name(s) to anonymize.
FAMILY_CONTEXT_PATTERNS = [
    # "Personne à prévenir : NAME" or "PERSONNE PREVENIR NAME" (various formats)
    r"[Pp][Ee]?[Rr][Ss][Oo][Nn][Nn][Ee]\s+[ÀàAa]?\s*[Pp][Rr][ÉéEe][Vv][Ee][Nn][Ii][Rr]\s*:?\s*\n?\s*([A-ZÀ-Üa-zà-ÿ][\w\sÀ-ÿ,\.\-]+?)(?=\s*\n\s*\n|\s*\n\s*[Pp]atient|\s*$)",
    # "Mère : NAME" / "Père : NAME" on same line (with actual name, not just "Mère")
    r"(?:Mère|Père|Parent)\s*:\s*([A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-,]{3,})(?=\s*[,\n]|\s+\d|\s*$)",
    # "accompagné par/de NAME"
    r"[Aa]ccompagn[ée]\s+(?:par|de)\s*:?\s*([A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ \-,]{3,})(?=\s*[,\n]|\s+\d|\s*$)",
]


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
    # Get the patient tag (all tokens map to the same tag)
    patient_tag = list(pseudonym_map.values())[0] if pseudonym_map else "[PATIENT]"
    for variant in patient_info["full_variants"]:
        pattern = re.escape(variant)
        matches = re.findall(pattern, result, re.IGNORECASE)
        if matches:
            result = re.sub(pattern, patient_tag, result, flags=re.IGNORECASE)
            stats["patient_name_multiword"] += len(matches)
            for m in matches:
                replacements.append({"original": m, "replacement": patient_tag, "category": "patient_name"})

    # --- Layer A2: Family/contact names (parents, guardians) ---
    # 1. Extract names from family context patterns (always PII)
    # 2. Split compound names: "TOURE SOULEMANE,CAMARA GNAMA" → 2 names
    # 3. Replace globally so they're caught everywhere in the dossier
    # Must run BEFORE Layer B so patient tokens inside family names
    # don't get replaced individually first.
    family_names = []
    for pattern in FAMILY_CONTEXT_PATTERNS:
        for match in re.finditer(pattern, result, re.MULTILINE):
            raw = match.group(1).strip()
            if not raw or len(raw) < 3:
                continue
            # Split compound names on comma/semicolon
            parts = re.split(r'[,;]\s*', raw)
            for part in parts:
                name = part.strip()
                if not name or len(name) < 3:
                    continue
                # Skip common role labels (not names)
                if name.lower() in {"mère", "père", "parent", "parents", "mère ",
                                    "tuteur", "tutrice", "oncle", "tante",
                                    "grand-mère", "grand-père", "soeur", "frère"}:
                    continue
                # Skip if entirely composed of patient tokens — Layer B handles those
                tokens_upper = {t.upper() for t in name.split()}
                if tokens_upper.issubset(patient_tokens_upper):
                    continue
                family_names.append(name)

    # Deduplicate, longest first, replace globally
    seen_family = set()
    for name in sorted(family_names, key=len, reverse=True):
        key = name.upper()
        if key in seen_family:
            continue
        seen_family.add(key)
        pat = re.escape(name)
        count = len(re.findall(pat, result, re.IGNORECASE))
        if count:
            result = re.sub(pat, "[ANONYMIZED]", result, flags=re.IGNORECASE)
            stats["family_contact"] += count
            replacements.append({"original": name, "replacement": "[ANONYMIZED]", "category": "family_contact"})
            logger.info(f"Layer A2: '{name}' → [ANONYMIZED] ({count}x)")

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
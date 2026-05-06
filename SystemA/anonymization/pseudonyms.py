import re
import unicodedata


def _normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_patient_map(patient_info: dict, patient_id: str = "001") -> dict[str, str]:
    tag = f"[PATIENT_{patient_id}]"
    mapping = {}

    for token in patient_info.get("all_tokens", []):
        if len(token) < 1:
            continue
        mapping[token] = tag
        mapping[token.upper()] = tag
        mapping[token.title()] = tag
        mapping[token.lower()] = tag

    # full firstname string, e.g. "U Teliane"
    if patient_info.get("firstnames"):
        fn = " ".join(patient_info["firstnames"])
        mapping[fn] = tag
        mapping[fn.upper()] = tag
        mapping[fn.title()] = tag
        mapping[fn.lower()] = tag

    # lastname + firstname full string
    if patient_info.get("lastnames") and patient_info.get("firstnames"):
        ln = " ".join(patient_info["lastnames"])
        fn = " ".join(patient_info["firstnames"])
        for full in [f"{ln} {fn}", f"{ln}, {fn}"]:
            mapping[full] = tag
            mapping[full.upper()] = tag
            mapping[full.title()] = tag
            mapping[full.lower()] = tag

    if len(patient_info.get("lastnames", [])) > 1:
        concat = "".join(patient_info["lastnames"])
        mapping[concat] = tag
        mapping[concat.upper()] = tag
        mapping[concat.title()] = tag
        mapping[concat.lower()] = tag

    return mapping


def assign_entity_tags(names: list[str], prefix: str) -> dict[str, str]:
    """
    Normalize names before assigning tags, so
    'Hôpital Robert Debré' and 'Hôpital ROBERT DEBRE'
    get the same entity tag.
    """
    tag_map = {}
    normalized_to_tag = {}
    idx = 1

    for name in names:
        key = _normalize_key(name)
        if not key:
            continue

        if key not in normalized_to_tag:
            normalized_to_tag[key] = f"[{prefix}_{idx:03d}]"
            idx += 1

        tag_map[name] = normalized_to_tag[key]

    return tag_map
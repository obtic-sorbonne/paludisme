"""
Pseudonym generation — replaces patient name tokens with [PATIENT_NNN].

Given a patient name dict and an ID, generates a mapping
{real_token: "[PATIENT_001]"} covering all case variants and
concatenated forms.
"""


def generate(patient_info: dict, patient_id: str = "001") -> dict[str, str]:
    """
    Build {real_token: replacement} map for all case variants.

    All tokens for a given patient map to the same tag, e.g. [PATIENT_001].
    """
    tag = f"[PATIENT_{patient_id}]"
    mapping = {}

    for ln in patient_info["lastnames"]:
        mapping[ln] = tag
        mapping[ln.title()] = tag
        mapping[ln.lower()] = tag

    for fn in patient_info["firstnames"]:
        mapping[fn] = tag
        mapping[fn.title()] = tag
        mapping[fn.lower()] = tag

    # Concatenated lastnames (e.g. NDOUNKEDJATCHEU)
    if len(patient_info["lastnames"]) > 1:
        real_concat = "".join(patient_info["lastnames"])
        mapping[real_concat] = tag
        mapping[real_concat.title()] = tag
        mapping[real_concat.lower()] = tag

    return mapping
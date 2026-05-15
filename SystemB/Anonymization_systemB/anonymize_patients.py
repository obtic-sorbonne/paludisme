#!/usr/bin/env python3
"""
anonymize_patients.py
Location: ~/digitize_medical_records/Anonymization_systemB/anonymize_patients.py

Fully config-driven anonymization. ALL rules come from anonymization_config.yaml.
Zero hardcoding - works for any institution/document type.

Usage:
  python anonymize_patients.py --patient RDB_0186 --id 001
  python anonymize_patients.py --all
  python anonymize_patients.py --all --config /path/to/other_config.yaml
"""

import re
import sys
import csv
import argparse
import logging
import unicodedata
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR    = "/home/lfarooq/digitize_medical_records"
OUTPUT_BASE = f"{WORK_DIR}/outputs/patients"
FINAL_DIR   = f"{WORK_DIR}/outputs/final_anonymized"
CONFIG_FILE = f"{WORK_DIR}/Anonymization_systemB/anonymization_config.yaml"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        log.info(f"Config loaded: {path}")
        return cfg
    except FileNotFoundError:
        log.warning(f"Config not found: {path} — using minimal defaults")
        return {"safe_words": [], "sensitive_fields": [], "pattern_detectors": {},
                "name_context_labels": [], "name_formats": {}, "free_text_name_triggers": [],
                "table_schemas": [], "institution": {"known_services": []}}
    except Exception as e:
        log.error(f"Config error: {e}")
        return {}


# ── Text normalization ─────────────────────────────────────────────────────────
def norm(s: str) -> str:
    """Normalize for comparison: strip accents, uppercase, collapse spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).upper().strip()


# ── Patient name extraction ────────────────────────────────────────────────────
def extract_patient_name(text: str) -> dict:
    """
    Extract patient name from merged text.
    Tries multiple patterns found in French hospital documents.
    Returns dict with lastname, firstname, variants list.
    """
    candidates = []

    # Pattern 1: "Nom: AZZOLA\nPrenom: KAREN"
    m1 = re.search(r"(?mi)^\s*Nom\s*:\s*([^\n]{1,60})\s*$", text)
    m2 = re.search(r"(?mi)^\s*Pr[ée]nom\s*:\s*([^\n]{1,60})\s*$", text)
    if m1 and m2:
        ln = m1.group(1).strip().split()[0]
        fn = m2.group(1).strip().split()[0]
        if len(ln) > 1 and len(fn) > 1:
            candidates.append((ln, fn))

    # Pattern 2: CRU style block "AZZOLA\nKAREN\n(75010)"
    m3 = re.search(r"(?m)^([A-ZÀ-Ü]{2,})\n([A-ZÀ-Ü][A-Za-zÀ-ÿ]{1,})\n\(\d{5}\)", text)
    if m3:
        candidates.append((m3.group(1), m3.group(2)))

    # Pattern 3: Lab style "Nom patient AZZOLA, KAREN"
    m4 = re.search(r"Nom\s+patient\s+([A-ZÀ-Ü]{2,}),\s*([A-ZÀ-Ü][A-Za-zÀ-ÿ]+)", text)
    if m4:
        candidates.append((m4.group(1), m4.group(2)))

    # Pattern 4: "de l'enfant AZZOLA, KAREN"
    m5 = re.search(r"de\s+l['']\s*enfant\s+([A-ZÀ-Ü]{2,}),?\s+([A-ZÀ-Ü][A-Za-zÀ-ÿ]+)", text)
    if m5:
        candidates.append((m5.group(1), m5.group(2)))

    if not candidates:
        return {}

    ln, fn = max(candidates, key=lambda c: len(c[0]) + len(c[1]))
    log.info(f"Patient identified: {ln} {fn}")

    variants = sorted(set(v for v in [
        f"{ln} {fn}", f"{ln}, {fn}", f"{fn} {ln}",
        f"{ln}", f"{fn}",
        f"{ln.title()}", f"{fn.title()}",
        f"{ln.lower()}", f"{fn.lower()}",
    ] if len(v) > 1), key=len, reverse=True)

    return {"lastname": ln, "firstname": fn, "variants": variants}


# ── Build name patterns from config ───────────────────────────────────────────
def _norm_label(s: str) -> str:
    """Normalize label for regex: strip accents, make pattern accent-insensitive."""
    import unicodedata
    # Build pattern that matches both accented and unaccented versions
    result = []
    for ch in s:
        nfkd = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in nfkd if not unicodedata.combining(c))
        if base != ch:
            # Has accent - match either version
            result.append(f"[{re.escape(base)}{re.escape(ch)}]")
        else:
            result.append(re.escape(ch))
    return "".join(result)


def build_name_patterns(config: dict) -> list:
    """
    Build regex patterns for staff/person names entirely from YAML config.
    No hardcoding - all patterns derived from config sections.
    """
    patterns = []
    name_re = r"[A-ZÀ-Ü][A-Za-zÀ-ÿ'\-]+"

    # 1. Name context labels from config (same-line)
    # IMPORTANT: pattern stops at newline to avoid capturing email on next line
    for label in config.get("name_context_labels", []):
        label_pat = _norm_label(label)
        # Match name on same line only - stop before newline or role suffix
        pat = (rf"(?i)(?<![A-Za-zÀ-ÿ]){label_pat}[\s\(\)]*[:\-]?\s*"
               rf"({name_re}(?:[,\s]+{name_re}){{0,4}}?)"
               rf"(?=\s*(?:,\s*(?:PH|CCA|MCU|PHU|AHU|HU|MD|PhD)\b|[\n\r]|$))")
        patterns.append(("context_label", pat))

    # 1b. Multi-line context labels: label on one line, name on next
    # e.g. "Validé par:\nDr. FENNETEAU, Odile"
    for label in config.get("multiline_context_labels", []):
        label_pat = _norm_label(label)
        pat = (rf"(?i)(?<![A-Za-zÀ-ÿ]){label_pat}[:\-]?[ \t]*"
               rf"\n[ \t]*(?:Dr\.|Pr\.|Dr |Pr |Docteur |Professeur )?"
               rf"({name_re}(?:[ \t]+{name_re}){{0,4}})"
               rf"[ \t]*(?:\n|$)")
        patterns.append(("multiline_context", pat))

    # 2. Name formats from config
    formats = config.get("name_formats", {})

    if formats.get("lastname_comma_firstname", True):
        # BITTAN, Jerome  or  FENNETEAU, Odile
        patterns.append(("lastname_comma", rf"\b([A-ZÀ-Ü]{{3,}},\s*[A-ZÀ-Ü]{name_re[1:]})\b"))

    if formats.get("allcaps_then_firstname", True):
        # MERCIER Jean Christophe
        patterns.append(("allcaps_firstname",
                         rf"\b([A-ZÀ-Ü]{{4,}}\s+[A-ZÀ-Ü][a-zà-ÿ]+(?:\s+[A-ZÀ-Ü][a-zà-ÿ]+)?)\b"))

    # 3. Free text triggers from config
    for trigger_def in config.get("free_text_name_triggers", []):
        phrase   = trigger_def.get("phrase", "")
        position = trigger_def.get("position", "before")
        if not phrase:
            continue
        # Allow whitespace including newlines between words of the phrase
        phrase_words = phrase.split()
        escaped_parts = [re.escape(w) for w in phrase_words]
        escaped = r"[\s\n]+".join(escaped_parts)
        if position == "before":
            # NAME [phrase] - allow name to span up to one newline
            pat = rf"({name_re}(?:[\s\n]+{name_re}){{0,2}})\s+{escaped}"
        else:
            # [phrase] NAME
            pat = rf"{escaped}\s+({name_re}(?:\s+{name_re}){{0,2}})"
        patterns.append(("free_text_trigger", pat))

    return patterns


# ── Qwen 7B NER ────────────────────────────────────────────────────────────────
def _run_qwen_ner(text: str, model: str, url: str, chunk_size: int) -> dict:
    """
    Use local Qwen 7B via ollama to extract person names from text.
    Processes text in chunks to handle long documents.
    Returns dict of {name: True} for all found person names.

    Qwen understands context so it correctly ignores:
    - Medical terms (Chlore, Creatinine, Coli...)
    - Drug names (Malarone, Halfan...)
    - Abbreviations (NFS, CRP...)
    - Lab values and measurements

    No safe_words needed for name detection - truly universal.
    """
    import urllib.request
    import json

    PROMPT_TEMPLATE = """You are a medical document anonymization assistant.
Extract ALL person names (doctors, nurses, patients, family members, any people) from the French medical text below.

Rules:
- Return ONLY real person names, one per line
- Include first names, last names, or full names
- Do NOT include: medical terms, drug names, lab values, abbreviations, institution names, city names
- Do NOT include words like: Chlore, Sodium, Hematocrite, Coli, Falciparum, Paris, Urgences
- If no names found, return: NONE

Medical text:
{chunk}

Person names found (one per line):"""

    all_names = {}
    # Split into overlapping chunks to avoid missing names at boundaries
    chunks = []
    for i in range(0, len(text), chunk_size - 200):
        chunks.append(text[i:i + chunk_size])

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        prompt = PROMPT_TEMPLATE.format(chunk=chunk)
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,      # deterministic
                "num_predict": 200,    # names list is short
                "top_p": 1,
            }
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "").strip()

            if response_text and response_text.upper() != "NONE":
                for line in response_text.splitlines():
                    name = line.strip().strip("-•*").strip()
                    # Basic validation: must look like a name
                    if (name and len(name) >= 3
                            and not name.upper().startswith("NONE")
                            and any(c.isalpha() for c in name)
                            and len(name) < 60):
                        all_names[name] = True

        except Exception as e:
            log.warning(f"Qwen NER chunk {i+1}/{len(chunks)} failed: {e}")
            continue

    return all_names


# ── Core anonymizer ────────────────────────────────────────────────────────────
def anonymize_text(text: str, patient_info: dict,
                   patient_id: str, config: dict) -> tuple:
    """
    Fully config-driven anonymization.
    Returns (anonymized_text, stats_dict, replacements_list)
    """
    safe_norm  = {norm(w) for w in config.get("safe_words", [])}
    services   = config.get("institution", {}).get("known_services", [])
    detectors  = config.get("pattern_detectors", {})
    matches    = []

    def add(start, end, tag, category):
        if start < end:
            matches.append({"start": start, "end": end,
                            "replacement": tag, "category": category,
                            "original": text[start:end]})

    def is_safe(name: str) -> bool:
        toks = set(norm(name).split())
        return bool(toks & safe_norm)

    # ── 1. Patient name ────────────────────────────────────────────────────
    patient_tag = f"[PATIENT_{patient_id}]"
    for variant in patient_info.get("variants", []):
        if is_safe(variant):
            continue
        for m in re.finditer(re.escape(variant), text, re.IGNORECASE):
            add(m.start(), m.end(), patient_tag, "patient_name")

    # ── 2. Built-in pattern detectors (toggled from config) ───────────────
    # All regex patterns defined here - toggled via YAML, never hardcoded logic
    BUILTIN = {
        "social_security_number": (
            r"\b([12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2})\b",
            "[ANONYMIZED]", "nir", 1),
        "hospital_patient_id": (
            r"\*(\d{8,20}[_\d]*)\*",
            "[ANONYMIZED]", "dossier_id", 1),
        "npi_identifier": (
            r"\bNPI\s*[: ]+\s*(\d{6,15})",
            "[ANONYMIZED]", "npi", 1),
        "sejour_id": (
            r"N[°o]?\s*de\s+s[ée]jour\s*[:\-]?\s*(\d{5,20})",
            "[ANONYMIZED]", "sejour_id", 1),
        "phone_number": (
            r"\b(0[1-9][\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2})\b",
            "[ANONYMIZED]", "telephone", 1),
        "email_address": (
            r"\b([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+)\b",
            "[ANONYMIZED]", "email", 1),
        "street_address": (
            r"(\d{1,4}\s*(?:bis|ter)?\s*(?:rue|avenue|boulevard|impasse|place|route|chemin|allee)\s+[^\n]{3,80})",
            "[ANONYMIZED]", "adresse", 1),
        "postal_code": (
            r"\b(\d{5}\s+[A-Za-z][A-Za-z0-9 \-]{2,30})\b",
            "[ANONYMIZED]", "adresse", 1),
        "finess_number": (
            r"N[°o]?\.?\s*F\.?I\.?N\.?E\.?S\.?S\.?\s*[:\-]?\s*(\d{6,15})",
            "[ANONYMIZED]", "finess", 1),
        "lab_request_id": (
            r"\bDemande\s+(\d{3}/\d{2}-\d{4,10})\b",
            "[ANONYMIZED]", "lab_request", 1),
    }

    for key, (pattern, tag, category, grp) in BUILTIN.items():
        if not detectors.get(key, True):
            continue
        try:
            for m in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                g = grp if m.lastindex and grp <= m.lastindex else 0
                add(m.start(g), m.end(g), tag, category)
        except re.error as e:
            log.warning(f"Pattern error ({key}): {e}")

    # ── 3. Sensitive field values (from config) ────────────────────────────
    for field in config.get("sensitive_fields", []):
        pat = rf"(?mi)^\s*{re.escape(field)}\s*[:\-]\s*([^\n]{{1,120}})\s*$"
        for m in re.finditer(pat, text):
            val = m.group(1).strip()
            if val and not is_safe(val):
                add(m.start(1), m.end(1), "[ANONYMIZED]", "sensitive_field")

    # ── 4. Institution services (from config) ──────────────────────────────
    for svc in services:
        for m in re.finditer(re.escape(svc), text, re.IGNORECASE):
            add(m.start(), m.end(), "[ANONYMIZED]", "institution")

    # ── 5. Staff/person names (fully from config patterns) ─────────────────
    # Assign consistent [STAFF_001] tags across entire document
    staff_registry: dict = {}
    staff_counter  = [0]

    def get_staff_tag(name: str) -> str:
        key = norm(name)
        if key not in staff_registry:
            staff_counter[0] += 1
            staff_registry[key] = f"[STAFF_{staff_counter[0]:03d}]"
        return staff_registry[key]

    name_patterns = build_name_patterns(config)
    for pat_type, pattern in name_patterns:
        try:
            for m in re.finditer(pattern, text, re.MULTILINE):
                name = m.group(1).strip(" ,;-")
                if not name or len(name) < 4:
                    continue
                if is_safe(name):
                    continue
                # Skip if looks like a section header (all caps, no spaces)
                if norm(name) == name.upper() and " " not in name:
                    continue
                tag = get_staff_tag(name)
                add(m.start(1), m.end(1), tag, "staff_name")
        except (re.error, IndexError) as e:
            log.warning(f"Name pattern error ({pat_type}): {e}")

    # ── 6. Qwen 7B NER - context-aware name extraction ────────────────────
    # Uses local Qwen 7B via ollama to find person names.
    # Understands context so it won't confuse "Chlore" or "Coli" with names.
    # No safe_words needed for name detection - fully universal.
    qwen_cfg     = config.get("qwen_ner", {})
    qwen_enabled = qwen_cfg.get("enabled", True)
    qwen_model   = qwen_cfg.get("model", "qwen2.5vl:7b")
    import os as _os
    _ollama_base = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    qwen_url     = qwen_cfg.get("url", f"{_ollama_base}/api/generate")
    chunk_size   = qwen_cfg.get("chunk_chars", 3000)

    if qwen_enabled:
        qwen_names = _run_qwen_ner(text, qwen_model, qwen_url, chunk_size)
        log.info(f"Qwen NER: found {len(qwen_names)} name candidates")
        for name, positions in qwen_names.items():
            name = name.strip().strip(".,;:-")
            # Minimum length and must contain at least one letter sequence of 3+
            if not name or len(name) < 4:
                continue
            # Must look like a real name - at least one word of 3+ letters
            words = [w for w in re.split(r"[\s,]+", name) if len(w) >= 3 and w.isalpha()]
            if not words:
                continue
            # Skip common French words that are NOT names
            common_words = {
                "mme", "mme.", "monsieur", "madame", "enfant", "patient",
                "service", "urgence", "pediatrie", "hopital", "medecin",
                "docteur", "interne", "senior", "equipe", "suite",
                "depuis", "avant", "apres", "ainsi", "cette", "dans",
                "avec", "sans", "pour", "mais", "bien", "tout", "plus",
                "lors", "sous", "type", "absence", "normale", "stase",
            }
            if norm(name) in {norm(w) for w in common_words}:
                continue
            # Skip if all words are short (likely abbreviations)
            if all(len(w) <= 3 for w in name.split()):
                continue
            # Skip patient name tokens
            if any(norm(v) == norm(name) or norm(name) in norm(v)
                   for v in patient_info.get("variants", [])):
                continue
            tag = get_staff_tag(name)
            # Build flexible pattern: match name with possible whitespace/newline between words
            # e.g. "Patricia Mariani" matches across line breaks
            # Also matches format variations (JEAN-CHRISTOPHE vs Jean Christophe)
            try:
                # Split name into words and join with flexible whitespace
                name_words = re.split(r"[\s,\-]+", name.strip())
                name_words = [w for w in name_words if w]
                if len(name_words) == 1:
                    # Single word: simple word-boundary match
                    pattern = (r"(?<![A-Za-zÀ-ÿ])" +
                               re.escape(name_words[0]) +
                               r"(?![A-Za-zÀ-ÿ])")
                else:
                    # Multi-word: allow flexible whitespace/newline between words
                    parts = [re.escape(w) for w in name_words]
                    pattern = (r"(?<![A-Za-zÀ-ÿ])" +
                               r"[\s\-,]*".join(parts) +
                               r"(?![A-Za-zÀ-ÿ])")
                for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    add(m.start(), m.end(), tag, "qwen_ner")
            except re.error:
                pass

    # ── Apply replacements ─────────────────────────────────────────────────
    final_text, applied = _apply_replacements(text, matches)

    stats = defaultdict(int)
    for r in applied:
        stats[r["category"]] += 1

    return final_text, dict(stats), applied


# ── Overlap resolution & application ──────────────────────────────────────────
def _apply_replacements(text: str, matches: list) -> tuple:
    if not matches:
        return text, []

    priority = {
        "patient_name": 0,
        "nir": 1, "sejour_id": 1,
        "telephone": 2, "email": 2, "npi": 2, "dossier_id": 2,
        "adresse": 3, "sensitive_field": 3,
        "family_contact": 4,
        "staff_name": 5, "spacy_person": 5,
        "institution": 6,
    }

    matches = sorted(matches, key=lambda m: (
        m["start"],
        priority.get(m["category"], 9),
        -(m["end"] - m["start"])
    ))

    kept = []
    for m in matches:
        conflict = next(
            (k for k in kept if m["start"] < k["end"] and m["end"] > k["start"]), None
        )
        if conflict is None:
            kept.append(m)
        else:
            mp = priority.get(m["category"], 9)
            kp = priority.get(conflict["category"], 9)
            if mp < kp or (mp == kp and (m["end"]-m["start"]) > (conflict["end"]-conflict["start"])):
                kept.remove(conflict)
                kept.append(m)

    kept.sort(key=lambda x: x["start"])

    out = text
    log_entries = []
    for m in sorted(kept, key=lambda x: x["start"], reverse=True):
        original = out[m["start"]:m["end"]]
        out = out[:m["start"]] + m["replacement"] + out[m["end"]:]
        log_entries.append({**m, "original": original})

    log_entries.reverse()
    return out, log_entries


# ── Main pipeline ──────────────────────────────────────────────────────────────

def get_or_assign_patient_num(patient_id_str: str) -> str:
    """
    Look up patient_id_str in the registry.
    If found, return the existing number.
    If not found, assign next sequential number, save, and return it.
    Registry: outputs/patient_registry.json
    """
    import json as _json
    registry_path = Path(WORK_DIR) / "outputs" / "patient_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = _json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}
    if patient_id_str in registry:
        return registry[patient_id_str]
    existing_nums = [int(v) for v in registry.values() if v.isdigit()]
    next_num = max(existing_nums, default=0) + 1
    patient_num = f"{next_num:03d}"
    registry[patient_id_str] = patient_num
    registry_path.write_text(_json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return patient_num

def anonymize_patient(patient_id_str: str, patient_num: str, config: dict) -> bool:
    patient_dir = Path(OUTPUT_BASE) / patient_id_str
    raw_file    = patient_dir / f"{patient_id_str}_patient_raw.txt"

    if not raw_file.exists():
        log.error(f"Raw file not found: {raw_file}")
        return False

    log.info(f"Anonymizing {patient_id_str} → patient_{patient_num}")
    text = raw_file.read_text(encoding="utf-8", errors="replace")

    patient_info = extract_patient_name(text)
    if not patient_info:
        log.warning("Could not extract patient name")
        patient_info = {"variants": []}

    anon_text, stats, replacements = anonymize_text(
        text, patient_info, patient_num, config
    )

    # ── Save anonymized file (next to raw) ────────────────────────────────
    anon_path = patient_dir / f"{patient_id_str}_patient_anonymized.txt"
    anon_path.write_text(anon_text, encoding="utf-8")

    # ── Save to central final_anonymized folder ────────────────────────────
    final_dir = Path(FINAL_DIR)
    final_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    final_copy = final_dir / f"patient_{patient_num}_anonymized.txt"
    shutil.copy(str(anon_path), str(final_copy))

    # ── Save replacement log ───────────────────────────────────────────────
    log_path = patient_dir / f"{patient_id_str}_replacements.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "original", "replacement", "start", "end"]
        )
        writer.writeheader()
        writer.writerows(replacements)

    # ── Save summary ───────────────────────────────────────────────────────
    summary_path = patient_dir / f"{patient_id_str}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Patient     : patient_{patient_num}\n")
        f.write(f"Folder      : {patient_id_str}\n")
        f.write(f"Source      : {raw_file}\n")
        if patient_info.get("lastname"):
            f.write(f"Name        : {patient_info['lastname']} {patient_info['firstname']}\n")
        f.write(f"\nStats:\n")
        for k, v in sorted(stats.items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal replacements: {len(replacements)}\n")

    log.info(f"  → {anon_path.name} ({len(replacements)} replacements)")
    log.info(f"  → {final_copy}")
    for k, v in sorted(stats.items()):
        log.info(f"     {k}: {v}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Config-driven anonymization for System B"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Patient folder ID e.g. RDB_0186")
    group.add_argument("--all", action="store_true", help="Process all patients")
    parser.add_argument("--id", default=None,
                        help="Patient number tag (auto-assigned with --all)")
    parser.add_argument("--config", default=CONFIG_FILE)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.patient:
        ok = anonymize_patient(args.patient, args.id or "001", config)
        sys.exit(0 if ok else 1)
    else:
        dirs = sorted(
            d for d in Path(OUTPUT_BASE).iterdir()
            if d.is_dir() and (d / f"{d.name}_patient_raw.txt").exists()
        )
        if not dirs:
            log.error(f"No patient raw files in {OUTPUT_BASE}")
            sys.exit(1)
        log.info(f"Found {len(dirs)} patients")
        failed = 0
        for d in dirs:
            _num = get_or_assign_patient_num(d.name)
            if not anonymize_patient(d.name, _num, config):
                failed += 1
        log.info(f"Done: {len(dirs)-failed} OK, {failed} failed")
        sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
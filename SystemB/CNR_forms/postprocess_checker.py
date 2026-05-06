#!/usr/bin/env python3
"""
postprocess_checker.py - v4 (CORRECTED)

Improvements over v3:
  1. REMOVED extract_j_checkpoints() entirely
     - It was hallucinating data when Qwen correctly extracted nothing
     - Empty checkpoints in the form should remain empty
  
  2. REMOVED Bandelettes propagation
     - There are 3 result options (positif Ag P falciparum, 
       positif Ag commun, Ag autres espèces)
     - Auto-updating is unsafe
  
  3. ADDED 2nde intention detection
     - When two "Traitement débuté le" appear, the second is 2nde intention
     - When two "Dose totale" appear, the second is 2nde intention
     - When two "Durée en jours" appear, the second is 2nde intention
     - When two "Effet indésirable" appear, the second is 2nde intention
  
  4. ADDED "Date de naissance" filter (privacy)
  
  5. ADDED "Plasmodium spp" / "J33" hallucination filter
  
  6. KEPT all v2 functionality (Goutte épaisse, Frottis, etc.)
"""

import os
import re
import argparse
from pathlib import Path


def load_glm_pages(glm_dir):
    pages = {}
    if not glm_dir or not os.path.exists(glm_dir):
        return pages
    for f in sorted(os.listdir(glm_dir)):
        if f.endswith(".txt"):
            m = re.search(r'page-(\d+)', f)
            num = int(m.group(1)) if m else 0
            with open(os.path.join(glm_dir, f), encoding="utf-8") as fh:
                pages[num] = fh.read()
    return pages


def get_all_text(pages):
    return "\n".join(pages.values())


def normalize(text):
    text = text.lower().strip()
    for old, new in [
        ("é","e"),("è","e"),("ê","e"),("ë","e"),
        ("à","a"),("â","a"),("ä","a"),
        ("î","i"),("ï","i"),("ô","o"),("ö","o"),
        ("ù","u"),("û","u"),("ü","u"),("ç","c"),
        ("'","'"),("'","'"),("®",""),("©",""),
    ]:
        text = text.replace(old, new)
    return text


# ── GLM Extractors ────────────────────────────────────────────────

def extract_goutte_epaisse(glm_text):
    """Extract Goutte épaisse value with multiple pattern attempts."""
    patterns = [
        r"[Gg]outte [eé]paisse\s*\n\s*([^\n]+)",
        r"[Gg]outte [eé]paisse[:\s]+([A-Za-zÀ-ÿ][^\n]*)",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, glm_text)
        if m:
            val = m.group(1).strip()
            val = val.lstrip("- *#").strip()
            
            if not val or len(val) < 3:
                continue
            
            # Reject noise patterns
            noise_terms = ["pour", "annuler", "si fait", "(", "préc", "next", 
                          "frottis", "mince", "%", "trophozoites", "schizontes",
                          "presence"]
            if any(n in val.lower() for n in noise_terms):
                continue
            
            if len(val) > 30:
                continue
            
            return val
    return None


def extract_reanimation(glm_text):
    """Extract dont réanimation/SI value."""
    patterns = [
        r"dont r[eé]animation/SI[:\s]+(\d+)",
        r"dont r[eé]animation/SI\s*\n\s*(\d+)",
        r"r[eé]animation/SI\s*[:\s]+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, glm_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_frottis_mince(glm_text):
    """Extract Frottis mince statut + density."""
    pattern1 = r"[Ff]rottis mince\s*\n([^\n]+)\s*\n([\d.,]+\s*%)\s*\n?(\d+)?"
    m = re.search(pattern1, glm_text)
    if m:
        statut = m.group(1).strip()
        density = m.group(2).strip()
        count = m.group(3).strip() if m.group(3) else ""
        if any(n in statut.lower() for n in ["annuler", "si fait", "non fait",
                                              "pour", "positif", "negatif"]):
            statut = None
        if statut and "presence" in normalize(statut):
            if count:
                return f"{statut} {density} {count}"
            return f"{statut} {density}"
    pattern2 = r"[Ff]rottis mince\s*\n([\d.,]+\s*%)"
    m2 = re.search(pattern2, glm_text)
    if m2:
        return m2.group(1).strip()
    return None


def extract_repulsifs_cutanes(glm_text):
    """Extract Répulsifs cutanés value."""
    pattern = r"[Rr][eé]pulsifs cutan[eé]s\s*\n([^\n\-\(]+)"
    m = re.search(pattern, glm_text)
    if m:
        val = m.group(1).strip()
        if val and len(val) > 2 and "annuler" not in val.lower() \
                and "moustiquaires" not in val.lower() \
                and "autres" not in val.lower():
            
            # Reject long noisy values
            if len(val) > 30:
                return None
            
            # Reject values with "X" or "indication"
            if re.search(r"\bX\b|indication", val, re.IGNORECASE):
                return None
            
            return val
    return None


def extract_nb_total_cpes(glm_text):
    """Extract tablet count from Nb total Cpés/J."""
    pattern = r"[Nn]b total Cp[eé]s/J\s+(\d+(?:\.\d+)?)"
    m = re.search(pattern, glm_text)
    if m:
        return m.group(1).strip()
    return None


# ── Hallucination filters ─────────────────────────────────────────

HALLUCINATED_LABELS = [
    # Empty checkboxes consistently misread as "Oui"
    "congénital", "congenital",
    "transfusionnel",
    "aéroportuaire", "aeroportuaire",
    "cryptique",
    "suspicion d'autochtone", "suspicion d autochtone",
    "accidentel (aes", "accidentel aes",
    
    # Patient identifiers (privacy)
    "primo-arrivé", "primo-arrive", "primo arrive",
    "prenom",
    "nom:",
    "id patient",
    "date de naissance",  # Privacy + always wrong
    
    # Chimioprophylaxie sub-fragments
    "chimioprophylaxie: oui",
    "chimioprophylaxie: malarone",
    "chimioprophylaxie: aux",
    "chimioprophylaxie: chez",
    
    # Document metadata
    "prélèvement effectué", "prelevement effectue",
    "envoyé le", "envoye le",
    "reconsulte pour",
    "densité parasitaire (µlitre)", "densite parasitaire (ulitre)",
    
    # Free-text fields
    "parasitémie", "parasitemie",
    "si non, y a-t-il", "si non y a-t-il", "si non y a t il",
    "a cette occasion une recherche",
    "troubles de la conscience minimes",
    "paludismes \"autochtones\" (adresser",
    
    # Plasmodium spp as label (it's a value)
    "plasmodium spp",
    
    # J33 hallucination (not a real checkpoint)
    "j33",
    "autre: j33",
    
    # Other noise
    "absence: absence",
    "perdu de vue",
    "validation senior",
]


def is_hallucinated(label, value):
    """Check if a label-value pair is a known hallucination."""
    ll = normalize(label)
    
    for h in HALLUCINATED_LABELS:
        h_norm = normalize(h)
        if h_norm in ll:
            return True
    
    # label=value duplicates
    if normalize(label) == normalize(value):
        return True
    
    # Date-only labels
    if re.match(r"^\d{1,2}[\s/]\d{1,2}[\s/]\d{2,4}$", label.strip()):
        return True
    
    # Just "Date:"
    if label.strip().lower() == "date":
        return True
    
    # "Traitement:" with long free-text description
    if normalize(label) == "traitement" and len(value) > 30:
        return True
    
    return False


def is_wrong_dose_totale(label, value, glm_text):
    """Fix Dose totale if Qwen got it wrong."""
    if "dose totale" not in normalize(label):
        return False, None
    glm_nb = extract_nb_total_cpes(glm_text)
    if glm_nb and glm_nb != value:
        return True, glm_nb
    return False, None


# ── 2nde intention detection ─────────────────────────────────────

# Fields that can appear twice (1ère intention + 2nde intention)
INTENTION_FIELDS = [
    "traitement débuté le",
    "dose totale",
    "durée en jours",
    "effet indésirable",
]


def relabel_2nde_intention(pred_pairs):
    """
    Detect duplicate fields and relabel the second occurrence as "2nde intention".
    
    Form structure:
        Traitement anti-palustre de 1ère intention: ...
        Traitement débuté le: Soir       ← 1st occurrence (1ère intention)
        Dose totale (mg/J): 400          ← 1st occurrence
        Durée en jours: 1                ← 1st occurrence
        
        Médicament de 2nde intention ou de relais: ...
        Traitement débuté le: Matin      ← 2nd occurrence → "2nde intention"
        Dose totale (mg/J): ...          ← 2nd occurrence
        Durée en jours: ...              ← 2nd occurrence
    """
    # Track which intention fields we've seen
    seen_count = {}
    new_pairs = []
    fixes = []
    
    for label, value in pred_pairs:
        ln = normalize(label)
        
        # Check if this is an intention field
        is_intention_field = False
        matched_field = None
        for field in INTENTION_FIELDS:
            if normalize(field) in ln and "2nde" not in ln:
                is_intention_field = True
                matched_field = field
                break
        
        if is_intention_field:
            count = seen_count.get(matched_field, 0)
            seen_count[matched_field] = count + 1
            
            if count == 1:
                # Second occurrence - relabel as 2nde intention
                # Preserve original casing where possible
                if matched_field == "traitement débuté le":
                    new_label = "Traitement débuté le 2nde intention"
                elif matched_field == "dose totale":
                    new_label = "Dose totale (mg/J) 2nde intention"
                elif matched_field == "durée en jours":
                    new_label = "Durée en jours 2nde intention"
                elif matched_field == "effet indésirable":
                    new_label = "Effet indésirable 2nde intention"
                else:
                    new_label = f"{label} 2nde intention"
                
                fixes.append((label, value, new_label))
                new_pairs.append((new_label, value))
                continue
        
        new_pairs.append((label, value))
    
    return new_pairs, fixes


# ── Main checker ──────────────────────────────────────────────────

def load_predictions(path):
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, _, value = line.partition(":")
            label, value = label.strip(), value.strip()
            if label and value:
                pairs.append((label, value))
    return pairs


def save_predictions(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for label, value in pairs:
            f.write(f"{label}: {value}\n")


def check_and_fix(pred_pairs, glm_text):
    result = []
    fixed = []
    added = []
    removed = []
    relabeled = []

    glm_goutte    = extract_goutte_epaisse(glm_text)
    glm_reanim    = extract_reanimation(glm_text)
    glm_frottis   = extract_frottis_mince(glm_text)
    glm_repulsifs = extract_repulsifs_cutanes(glm_text)
    glm_nb_cpes   = extract_nb_total_cpes(glm_text)

    found_goutte    = False
    found_reanim    = False
    found_frottis   = False
    found_repulsifs = False

    for label, value in pred_pairs:
        ln = normalize(label)

        # Remove known hallucinations
        if is_hallucinated(label, value):
            removed.append((label, value))
            print(f"  [REMOVE] Hallucinated: {label}: {value}")
            continue

        # Fix wrong dose totale
        wrong_dose, correct_dose = is_wrong_dose_totale(label, value, glm_text)
        if wrong_dose:
            fixed.append((label, value, correct_dose))
            result.append((label, correct_dose))
            print(f"  [FIX] Dose totale: {value} → {correct_dose}")
            continue

        # Track found fields
        if "goutte epaisse" in ln:
            found_goutte = True
            # Validate value (reject "Trophozoites/schizontes")
            if "trophozoit" in normalize(value) or "schizont" in normalize(value):
                if glm_goutte:
                    fixed.append((label, value, glm_goutte))
                    result.append((label, glm_goutte))
                    print(f"  [FIX] Goutte épaisse: {value} → {glm_goutte}")
                else:
                    removed.append((label, value))
                    print(f"  [REMOVE] Bad Goutte épaisse value: {value}")
                continue
            
            if glm_goutte and normalize(glm_goutte) != normalize(value):
                fixed.append((label, value, glm_goutte))
                result.append((label, glm_goutte))
            else:
                result.append((label, value))

        elif "reanimation" in ln or "réanimation" in ln:
            found_reanim = True
            if glm_reanim and normalize(glm_reanim) != normalize(value):
                fixed.append((label, value, glm_reanim))
                result.append((label, glm_reanim))
            else:
                result.append((label, value))

        elif "frottis mince" in ln:
            found_frottis = True
            if glm_frottis and normalize(glm_frottis) != normalize(value):
                fixed.append((label, value, glm_frottis))
                result.append((label, glm_frottis))
            else:
                result.append((label, value))

        elif "repulsifs cutanes" in ln or "répulsifs cutanés" in ln:
            found_repulsifs = True
            # Validate Qwen value
            if len(value) > 30 or "indication" in normalize(value):
                if glm_repulsifs:
                    fixed.append((label, value, glm_repulsifs))
                    result.append((label, glm_repulsifs))
                else:
                    removed.append((label, value))
                continue
            
            if glm_repulsifs and normalize(glm_repulsifs) != normalize(value):
                fixed.append((label, value, glm_repulsifs))
                result.append((label, glm_repulsifs))
            else:
                result.append((label, value))

        else:
            result.append((label, value))

    # Add missing fields from GLM
    if not found_goutte and glm_goutte:
        added.append(("Goutte épaisse", glm_goutte))
        result.append(("Goutte épaisse", glm_goutte))

    if not found_reanim and glm_reanim:
        added.append(("dont réanimation/SI", glm_reanim))
        result.append(("dont réanimation/SI", glm_reanim))

    if not found_frottis and glm_frottis:
        added.append(("Frottis mince", glm_frottis))
        result.append(("Frottis mince", glm_frottis))

    if not found_repulsifs and glm_repulsifs:
        added.append(("Répulsifs cutanés", glm_repulsifs))
        result.append(("Répulsifs cutanés", glm_repulsifs))

    # Relabel 2nde intention duplicates
    result, intention_fixes = relabel_2nde_intention(result)
    for old_label, value, new_label in intention_fixes:
        relabeled.append((old_label, value, new_label))
        print(f"  [RELABEL] '{old_label}' → '{new_label}' (2nde intention)")

    return result, fixed, added, removed, relabeled


def process_doc(pred_path, glm_dir, out_path=None, verbose=True):
    doc_id = Path(pred_path).stem

    pred_pairs = load_predictions(pred_path)
    if not pred_pairs:
        print(f"[SKIP] No predictions: {pred_path}")
        return

    pages = load_glm_pages(glm_dir)
    if not pages:
        print(f"[SKIP] No GLM text: {glm_dir}")
        return

    glm_text = get_all_text(pages)
    updated, fixed, added, removed, relabeled = check_and_fix(pred_pairs, glm_text)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {doc_id}")
        print(f"{'='*55}")
        if fixed:
            print(f"  ✅ FIXED ({len(fixed)}):")
            for label, old_val, new_val in fixed:
                print(f"     {label}")
                print(f"       Old : {old_val}")
                print(f"       New : {new_val}")
        if relabeled:
            print(f"  🏷️  RELABELED ({len(relabeled)}) for 2nde intention:")
            for old_label, val, new_label in relabeled:
                print(f"     {old_label} → {new_label}")
        if added:
            print(f"  ➕ ADDED ({len(added)}):")
            for label, val in added:
                print(f"     {label}: {val}")
        if removed:
            print(f"  🗑  REMOVED ({len(removed)}) hallucinations:")
            for label, val in removed:
                print(f"     {label}: {val}")
        if not fixed and not added and not removed and not relabeled:
            print(f"  ✓ No corrections needed")

    out = out_path or pred_path
    save_predictions(out, updated)
    if verbose:
        print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=None)
    parser.add_argument("--glm_dir", default=None)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--glm_base", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    if args.pred and args.glm_dir:
        process_doc(args.pred, args.glm_dir, out_path=args.out)

    elif args.pred_dir and args.glm_base:
        pred_files = sorted(f for f in os.listdir(args.pred_dir)
                           if f.endswith(".txt") and
                           not f.endswith("_checked.txt") and
                           not f.endswith("_original.txt"))
        for fname in pred_files:
            doc_id = os.path.splitext(fname)[0]
            pred_path = os.path.join(args.pred_dir, fname)
            glm_dir = os.path.join(args.glm_base, doc_id, "text")
            if not os.path.exists(glm_dir):
                print(f"[SKIP] No GLM dir for {doc_id}")
                continue
            out_path = pred_path if args.inplace else \
                       pred_path.replace(".txt", "_checked.txt")
            process_doc(pred_path, glm_dir, out_path=out_path)
    else:
        print("Provide either --pred + --glm_dir, or --pred_dir + --glm_base")
        parser.print_help()


if __name__ == "__main__":
    main()
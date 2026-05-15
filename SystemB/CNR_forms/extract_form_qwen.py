#!/usr/bin/env python3
"""
extract_form_qwen.py

Uses Qwen2.5-VL (72b by default) via Ollama to extract form fields,
then cross-checks against GLM-OCR text to fix known visual errors.

100% local — no data leaves your machine.

Usage:
    python extract_form_qwen.py \
        --img_dir /home/lfarooq/glm_ocr_runs/DOC_00184/pages \
        --glm_dir /home/lfarooq/glm_ocr_runs/DOC_00184/text \
        --out /home/lfarooq/digitize_medical_records/evaluation/predictions/DOC_00184.txt

    # Or from PDF directly:
    python extract_form_qwen.py \
        --pdf /path/to/DOC_00184.pdf \
        --glm_dir /home/lfarooq/glm_ocr_runs/DOC_00184/text \
        --out /home/lfarooq/digitize_medical_records/evaluation/predictions/DOC_00184.txt
"""

import os, sys, argparse, tempfile, time, re
from pathlib import Path
from PIL import Image
import ollama
import os as _os_env

MODEL = "qwen2.5vl:72b"

PROMPT = """This is a scanned French medical form about malaria (CNR Paludisme).

TASK: Extract every field where an answer is VISUALLY selected or filled in.

HOW TO READ THIS FORM:
- Radio buttons: a FILLED/DARK circle = selected. An EMPTY/HOLLOW circle = NOT selected.
  There is exactly ONE filled circle per question group.
- Checkboxes: a square with a TICK or FILL inside = checked. An EMPTY square = NOT checked.
- Dropdowns: text shown inside a bordered box = the selected value.
- Text fields: typed text inside a box = the value.

CRITICAL RULES:
1. Output format: Label: Value (one per line, no markdown, no bold, no bullets)
2. NEVER output "Selected", "Empty", "Ticked", "Checked" as values — output the actual option text
3. For radio buttons: output the TEXT of the filled option e.g. "Non" not "filled circle"
4. For checkboxes that are ticked: output Label: Oui
5. For "Durée du séjour" — ONLY output if one circle is CLEARLY darker than all others. If uncertain, SKIP.
6. For "Résidence durant le séjour en zone d'endémie" — options are: Urbain strict / Rural / Itinérant / Mixte / NSP. Look very carefully at which circle is filled.
7. For "Etat clinique au moment du diagnostic" — options are: Accès simple sans vomissements / Accès simple AVEC vomissements / Formes Asymptomatiques / Accès GRAVE / Paludisme Viscéral évolutif. Output only the ONE with a filled circle.
8. For "positif Ag P falciparum" checkbox if ticked → "Bandelettes résultat: positif Ag P falciparum"
9. For Chimioprophylaxie: first output the radio (Oui/Non/NSP), then separately each drug with its frequency
10. For J3/J4/J28 parasitology table: output "J3 ou J4 Parasitologie: Absence" etc. with full label
11. Skip: Annuler, voozanoo, Paris-Robert, Accueil, Déconnecter, page numbers, fiche_correspondant
12. Skip: Le symbole, https://, Loi n°, Le Centre National, Validation senior, Pages visitées
13. Skip: Sang, Plasma, Buvard, Prélèvement effectué, Commentaires & Remarques, Perdu de vue
14. Skip: ID patient, Nom, Prénom (patient identifiers)

Output ONLY filled fields, one per line: Label: Value"""

# ── Noise filter ─────────────────────────────────────────────────
NOISE = [
    "annuler", "voozanoo", "paris-robert", "fiche_correspondant",
    "le symbole", "https://", "http://", "loi n", "le centre national",
    "cnr paludisme", "precedent", "pages visit", "validation senior",
    "jysiriez", "si militaire", "sang:", "plasma:", "buvard:",
    "prelèvement", "prelevement", "commentaires & remarques",
    "label: value", "**label", "perdu de vue", "id patient",
    "here are", "the following", "filled/selected",
    "absence: absence", "perdu de vue: perdu",
]

def is_noise(line):
    ll = line.lower().strip()
    return any(n in ll for n in NOISE) or len(ll) < 4

def clean(raw):
    seen = set()
    out = []
    for line in raw.splitlines():
        line = line.strip()
        line = line.lstrip("- *#").strip()
        line = line.replace("**", "").strip()

        if not line or ":" not in line:
            continue
        if is_noise(line):
            continue

        label = line.split(":")[0].strip()
        value = line.partition(":")[2].strip()
        value = value.replace("**", "").strip()

        # Replace "Checked" with "Oui"
        if value.lower() in ["checked", "ticked", "selected", "coché"]:
            value = "Oui"

        # Skip empty or placeholder values
        if not value or value.lower() in ["", "-", "—", "/", "(empty)",
                                           "(selected)", "(ticked)", "nr"]:
            continue

        # Skip short labels
        if len(label) < 4:
            continue

        # Skip date-only labels (14/03/2007 etc)
        if re.match(r"^\d{2}/\d{2}/\d{4}$", label):
            continue

        final_line = f"{label}: {value}"
        if final_line not in seen:
            seen.add(final_line)
            out.append(final_line)
    return out


# ── GLM-OCR cross-check ───────────────────────────────────────────

# Fields where Qwen frequently makes visual errors
# We use GLM-OCR text to verify/correct these
VERIFY_FIELDS = {
    "résidence durant le séjour en zone d'endémie": {
        "options": ["Urbain strict", "Rural", "Itinérant / Mixte", "NSP"],
        "glm_search": ["Urbain strict", "Rural", "Itinérant / Mixte", "NSP"]
    },
    "etat clinique au moment du diagnostic": {
        "options": [
            "Accès simple sans vomissements",
            "Accès simple AVEC vomissements",
            "Formes Asymptomatiques",
            "Accès GRAVE",
            "Paludisme Viscéral évolutif"
        ],
        "glm_search": [
            "Accès simple sans vomissements",
            "Accès simple AVEC vomissements",
            "Formes Asymptomatiques",
            "Accès GRAVE",
            "Paludisme Viscéral"
        ]
    },
}

# Fields to always remove (consistently hallucinated)
ALWAYS_REMOVE = [
    "durée du séjour",
    "id patient",
    "nom:",
    "prénom:",
    "absence: absence",
    "perdu de vue",
    "validation senior",
]

def load_glm_text(glm_dir):
    """Load all GLM-OCR text files into one combined string."""
    if not glm_dir or not os.path.exists(glm_dir):
        return ""
    combined = []
    for f in sorted(os.listdir(glm_dir)):
        if f.endswith(".txt"):
            with open(os.path.join(glm_dir, f), encoding="utf-8") as fh:
                combined.append(fh.read())
    return "\n".join(combined)

def find_selected_in_glm(glm_text, options):
    """
    Given GLM-OCR text and a list of options,
    find which option appears to be selected.
    
    In GLM-OCR output, selected radio buttons appear as:
    '- Option text' or '@ Option text' or just the option on its own line
    after the question.
    
    We look for the option that appears with a selection marker,
    or as the first/only option on a line by itself.
    """
    if not glm_text:
        return None

    # Look for each option in the GLM text
    # GLM tends to output selected option differently — 
    # for radio groups it outputs ALL options but selected one
    # appears on a line starting with filled circle marker
    for option in options:
        # Check if option appears after a filled marker
        patterns = [
            rf"@\s*{re.escape(option)}",       # @ Option
            rf"◉\s*{re.escape(option)}",        # ◉ Option  
            rf"•\s*{re.escape(option)}",        # • Option
        ]
        for pat in patterns:
            if re.search(pat, glm_text, re.IGNORECASE):
                return option

    return None

def postprocess(lines, glm_text):
    """
    Post-process extracted lines:
    1. Remove always-hallucinated fields
    2. Cross-check visually unreliable fields against GLM-OCR
    3. Fix formatting issues
    """
    result = []

    for line in lines:
        label_lower = line.split(":")[0].strip().lower()

        # Remove always-hallucinated fields
        skip = False
        for bad in ALWAYS_REMOVE:
            if bad in label_lower:
                print(f"  [POST] Removed hallucinated field: {line}", flush=True)
                skip = True
                break
        if skip:
            continue

        # Cross-check visually unreliable fields
        corrected = False
        for field_key, field_info in VERIFY_FIELDS.items():
            if field_key in label_lower:
                glm_answer = find_selected_in_glm(glm_text, field_info["glm_search"])
                qwen_value = line.partition(":")[2].strip()

                if glm_answer and glm_answer.lower() != qwen_value.lower():
                    label = line.split(":")[0].strip()
                    corrected_line = f"{label}: {glm_answer}"
                    print(f"  [POST] Corrected via GLM-OCR:", flush=True)
                    print(f"         Qwen said : {line}", flush=True)
                    print(f"         GLM says  : {corrected_line}", flush=True)
                    result.append(corrected_line)
                    corrected = True
                elif glm_answer is None:
                    # GLM couldn't determine it either — keep Qwen's answer
                    print(f"  [POST] GLM couldn't verify '{field_key}', keeping Qwen: {line}", flush=True)
                break

        if not corrected:
            result.append(line)

    return result


# ── Image handling ────────────────────────────────────────────────

def to_jpeg(png_path):
    """Convert PNG to JPEG — fixes Qwen2.5-VL PNG crash bug."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.open(png_path).convert("RGB")
    img.save(tmp.name, "JPEG", quality=85)
    return tmp.name

def query_qwen(image_path, model=MODEL):
    if image_path.lower().endswith(".png"):
        image_path = to_jpeg(image_path)
    _ollama_host = _os_env.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    _client = ollama.Client(host=_ollama_host)
    response = _client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}]
    )
    return response["message"]["content"]

def get_pages(img_dir):
    exts = {".png", ".jpg", ".jpeg"}
    pages = []
    for f in sorted(os.listdir(img_dir)):
        p = Path(f)
        if p.suffix.lower() in exts and "page" in f.lower():
            pages.append(os.path.join(img_dir, f))
    return pages

def pdf_to_images(pdf_path, out_dir, dpi=150):
    try:
        import fitz
    except ImportError:
        print("ERROR: pip install pymupdf")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        p = os.path.join(out_dir, f"page-{i+1}.png")
        pix.save(p)
        paths.append(p)
        print(f"  Converted page {i+1}/{len(doc)}", flush=True)
    doc.close()
    return paths


# ── Main processing ───────────────────────────────────────────────

def process(img_dir, glm_dir, out_path, model="qwen2.5vl:72b"):
    pages = get_pages(img_dir)
    if not pages:
        print(f"ERROR: No page images in {img_dir}")
        sys.exit(1)

    # Load GLM-OCR text for cross-checking
    glm_text = load_glm_text(glm_dir)
    if glm_text:
        print(f"Loaded GLM-OCR text ({len(glm_text)} chars) for cross-checking\n")
    else:
        print("WARNING: No GLM-OCR text found — skipping cross-check\n")

    print(f"Processing {len(pages)} pages with {MODEL}...\n")
    all_lines = []

    for img_path in pages:
        name = Path(img_path).stem
        print(f"\n── {name} ──────────────────────────", flush=True)
        t0 = time.time()
        try:
            raw = query_qwen(img_path, model)
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] RAW:\n{raw}\n", flush=True)
            lines = clean(raw)
            print(f"  CLEANED ({len(lines)} fields):", flush=True)
            for l in lines:
                print(f"    {l}", flush=True)
            all_lines.extend(lines)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    # Post-process with GLM cross-check
    print(f"\n── POST-PROCESSING ──────────────────────────", flush=True)
    all_lines = postprocess(all_lines, glm_text)

    # Final dedup
    seen = set()
    final = []
    for line in all_lines:
        if line not in seen:
            seen.add(line)
            final.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final) + "\n")

    print(f"\n{'='*50}")
    print(f"DONE — {len(final)} fields extracted")
    print(f"Saved -> {out_path}")
    print("="*50)
    print("\nFINAL OUTPUT:")
    for l in final:
        print(f"  {l}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--glm_dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--model", default="qwen2.5vl:72b")
    args = parser.parse_args()
    model = args.model

    if args.pdf:
        tmp = os.path.join(
            os.path.dirname(os.path.abspath(args.out)),
            Path(args.pdf).stem + "_pages"
        )
        print(f"Converting PDF to images in {tmp}")
        pdf_to_images(args.pdf, tmp, args.dpi)
        img_dir = tmp
    elif args.img_dir:
        img_dir = args.img_dir
    else:
        print("ERROR: provide --img_dir or --pdf")
        sys.exit(1)

    process(img_dir, args.glm_dir, args.out, model)


if __name__ == "__main__":
    main()
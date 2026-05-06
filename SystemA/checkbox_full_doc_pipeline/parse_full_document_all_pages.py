from pathlib import Path
import argparse
import json
import re

import cv2

from parse_autres_techniques_visual import parse_autres_techniques_visual
from parse_page4_controle_parasito_visual import (
    load_json as load_controle_visual_json,
    extract_page_words as extract_controle_visual_words,
    parse_controle_block as parse_controle_block_visual,
)
from parse_circle_radio_visual_from_ocr_json import (
    load_json as load_circle_json,
    extract_page_words as extract_circle_words,
    parse_circle_radio_field,
)
from parse_radio_visual_from_ocr_json import (
    load_json as load_visual_json,
    extract_page_words as extract_visual_words,
    parse_radio_field,
)
from parse_page3_from_ocr_text import (
    page3_specs,
    parse_examens_parasitologiques_block,
    parse_protection_personnelle_block,
    parse_chimioprophylaxie_block,
    parse_bandelettes_block,
    parse_page3_lab_values_block,
)
from cnr_common import (
    load_ocr_txt,
    clean_text,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    norm,
    compact_norm,
    parse_prefix_and_text,
    find_first_line_index,
)
from parse_page1_from_ocr_text import page1_specs
from parse_page2_from_ocr_text import page2_specs, parse_page2_clinical_block
from parse_page4_from_ocr_text import (
    page4_specs,
    parse_controle_parasitologique,
    parse_page4_treatment_block,
)


def extract_all_pages(lines: list[str]) -> list[dict]:
    pages = []
    current_page = None
    current_lines = []
    page_pat = re.compile(r"^===== PAGE (\d+) =====$")

    for raw in lines:
        line = clean_text(raw)
        m = page_pat.match(line)
        if m:
            if current_page is not None:
                pages.append(
                    {
                        "page_num": current_page,
                        "lines": [clean_text(x) for x in current_lines if clean_text(x)],
                    }
                )
            current_page = int(m.group(1))
            current_lines = []
        else:
            current_lines.append(raw)

    if current_page is not None:
        pages.append(
            {
                "page_num": current_page,
                "lines": [clean_text(x) for x in current_lines if clean_text(x)],
            }
        )

    return pages


def preprocess_page_lines(page_num: int, lines: list[str]) -> list[str]:
    replacements = [
        ("Afriain", "Africain"),
        ("Ethnicite", "Ethnicité"),
        ("Duree du sejour", "Durée du séjour"),
        ("endemie", "endémie"),
        ("adresse parun", "adressé par un"),
        ("recherchebiologiquede", "recherche biologique de"),
        ("Paludismea-t-elle", "Paludisme a-t-elle"),
        ("Acette", "A cette"),
        ("NoN", "NON"),
        ("Si OUl,resultat", "Si OUI, résultat"),
        ("Si OUl, resultat", "Si OUI, résultat"),
        ("Si oul,a quelle date", "Si OUI, à quelle date"),
        ("Paysderésidence", "Pays de résidence"),
        ("Paysd'endémie", "Pays d'endémie"),
        ("Datedepart", "Date départ"),
        ("Datede naissance", "Date de naissance"),
        ("Fréquence des sejours", "Fréquence des séjours"),
        ("Residence durant", "Résidence durant"),
        ("acces", "accès"),
        ("ONSP", "O NSP"),
        ("Povale", "P ovale"),
        ("Pvivax", "P vivax"),
        ("XHospitalisation", "X Hospitalisation"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
        ("O UI", "OUI"),
        ("O U I", "OUI"),
        ("demière", "dernière"),
        ("Symptomes", "Symptômes"),
        ("OAccès", "O Accès"),
        ("OMilitaires", "O Militaires"),
    ]

    out = []
    for line in lines:
        line = clean_text(line)
        for old, new in replacements:
            line = line.replace(old, new)
        out.append(line)

    return postprocess_lines(out)


def get_page_json_and_image_paths(ocr_txt_path: Path, page_num: int):
    stem = ocr_txt_path.stem
    json_path = Path(
        f"/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/{stem}_{page_num - 1}_{page_num - 1}_res.json"
    )
    image_path = Path(
        f"/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages/{stem}-{page_num}.png"
    )
    return json_path, image_path


VISUAL_FIELD_CONFIG = {
    "Consultation avant": {"field_key": "consultation_avant"},
    "Résidence en zone d'endémie": {"field_key": "residence_zone_endemie"},
    "Etat clinique au moment du diagnostic": {"field_key": "etat_clinique"},
    "Chimioprophylaxie utilisée": {"field_key": "chimioprophylaxie_oui_non_nsp"},
    "Evolution clinique": {"field_key": "evolution_clinique"},
    "Sexe": {"field_key": "sexe"},
}

CIRCLE_VISUAL_FIELD_CONFIG = {
    "Sexe": {"field_key": "sexe"},
    "Ethnicité": {"field_key": "ethnicite"},
    "Autres pays d'endémie": {"field_key": "autres_pays_endemie"},
    "Résidence en zone d'endémie": {"field_key": "residence_zone_endemie"},
    "Fréquence des séjours": {"field_key": "frequence_sejours"},
    "Patient adressé": {"field_key": "patient_adresse"},
    "Consultation avant": {"field_key": "consultation_avant"},
    "Etat clinique au moment du diagnostic": {"field_key": "etat_clinique"},
    "Antécédents de paludisme dans les 3 derniers mois": {"field_key": "antecedents_paludisme_3m"},
    "Femme enceinte ou parturiente": {"field_key": "femme_enceinte"},
    "Immunodépression connue": {"field_key": "immunodepression_connue"},
    "Paludismes autochtones": {"field_key": "paludismes_autochtones"},
    "Lame transmise par autre Labo": {"field_key": "lame_transmise"},
    "Protection Personnelle Anti-Moustiques": {"field_key": "protection_personnelle_oui_non_nsp"},
    "Chimioprophylaxie utilisée": {"field_key": "chimioprophylaxie_oui_non_nsp"},
    "Arrêt de la prise suite à intolérance/effet(s) secondaire(s)": {
        "field_key": "arret_intolerance_effets_secondaires"
    },
    "Utilisation traitement à visée Curative du paludisme dans les 30 derniers jours": {
        "field_key": "utilisation_traitement_curatif_30j"
    },
    "Prise en charge": {"field_key": "prise_en_charge"},
    "Evolution clinique": {"field_key": "evolution_clinique"},
    "Contrôle parasitologique P falciparum": {"field_key": "controle_parasitologique_overall"},
}


def _all_existing_page_assets(ocr_txt_path: Path):
    pages = []
    for page_num in range(1, 50):
        json_path, image_path = get_page_json_and_image_paths(ocr_txt_path, page_num)
        if json_path.exists() and image_path.exists():
            pages.append((page_num, json_path, image_path))
    return pages


def _visual_conf_rank(conf: str | None) -> int:
    if conf == "high":
        return 3
    if conf == "medium":
        return 2
    if conf == "low":
        return 1
    return 0


def run_visual_fallback_for_field(ocr_txt_path: Path, field_name: str):
    cfg = VISUAL_FIELD_CONFIG.get(field_name)
    if not cfg:
        return None

    field_key = cfg["field_key"]
    candidates = []

    for page_num, json_path, image_path in _all_existing_page_assets(ocr_txt_path):
        try:
            data = load_visual_json(json_path)
            data = data.get("res", data)
            words = extract_visual_words(data, page_num=1)

            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue

            debug_dir = Path(
                "/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/visual_debug"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

            page_stem = f"{ocr_txt_path.stem}_page{page_num}"
            result = parse_radio_field(words, img_bgr, field_key, debug_dir, page_stem)
            selected = result.get("selected_option")
            if selected:
                candidates.append(
                    {
                        "page_num": page_num,
                        "selected_option": selected,
                        "confidence": result.get("confidence"),
                        "raw_result": result,
                    }
                )
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (_visual_conf_rank(x.get("confidence")), -x["page_num"]),
        reverse=True,
    )
    return candidates[0]


def run_circle_visual_fallback_for_field(ocr_txt_path: Path, field_name: str):
    cfg = CIRCLE_VISUAL_FIELD_CONFIG.get(field_name)
    if not cfg:
        return None

    field_key = cfg["field_key"]
    candidates = []

    for page_num, json_path, image_path in _all_existing_page_assets(ocr_txt_path):
        try:
            data = load_circle_json(json_path)
            data = data.get("res", data)
            words = extract_circle_words(data, page_num=1)

            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue

            debug_dir = Path(
                "/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/circle_visual_debug"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

            page_stem = f"{ocr_txt_path.stem}_page{page_num}"
            result = parse_circle_radio_field(
                words,
                img_bgr,
                field_key,
                debug_dir=debug_dir,
                page_stem=page_stem,
            )
            selected = result.get("selected_option")
            if selected:
                candidates.append(
                    {
                        "page_num": page_num,
                        "selected_option": selected,
                        "confidence": result.get("confidence"),
                        "raw_result": result,
                    }
                )
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (_visual_conf_rank(x.get("confidence")), -x["page_num"]),
        reverse=True,
    )
    return candidates[0]


def run_controle_parasito_visual_fallback(ocr_txt_path: Path):
    candidates = []

    for page_num, json_path, image_path in _all_existing_page_assets(ocr_txt_path):
        try:
            data = load_controle_visual_json(json_path)
            data = data.get("res", data)
            words = extract_controle_visual_words(data, page_num=1)
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                return None

            result = parse_controle_block_visual(words, img_bgr=img_bgr)
            
            if not result or not result.get("found"):
                continue

            overall = run_circle_visual_fallback_for_field(
                ocr_txt_path, "Contrôle parasitologique P falciparum"
            )
            if overall and overall.get("selected_option"):
                result["control_overall"] = overall["selected_option"]
                result["overall_source"] = "circle_visual_fallback"
                result["overall_page_num"] = overall["page_num"]

            candidates.append({"page_num": page_num, "result": result})
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["page_num"], reverse=True)
    return candidates[0]["result"]


def run_autres_techniques_visual_fallback(ocr_txt_path: Path):
    candidates = []

    for page_num, json_path, image_path in _all_existing_page_assets(ocr_txt_path):
        try:
            data = load_circle_json(json_path)
            data = data.get("res", data)
            words = extract_circle_words(data, page_num=1)

            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue

            result = parse_autres_techniques_visual(words, img_bgr)
            if result:
                candidates.append({"page_num": page_num, "result": result})
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["page_num"], reverse=True)
    return candidates[0]["result"]


def apply_visual_fallbacks(ocr_txt_path: Path, merged_fields: dict) -> dict:
    out = {}

    for field, fr in merged_fields.items():
        item = dict(fr)
        selected = item.get("selected_options", [])
        found = item.get("found", False)
        should_try_visual = (not selected) or (not found)

        if should_try_visual:
            visual_selected = None
            visual_page = None
            visual_conf = None

            if field in VISUAL_FIELD_CONFIG:
                res = run_visual_fallback_for_field(ocr_txt_path, field)
                if res:
                    visual_selected = res.get("selected_option")
                    visual_page = res.get("page_num")
                    visual_conf = res.get("confidence")

            if not visual_selected and field in CIRCLE_VISUAL_FIELD_CONFIG:
                res = run_circle_visual_fallback_for_field(ocr_txt_path, field)
                if res:
                    visual_selected = res.get("selected_option")
                    visual_page = res.get("page_num")
                    visual_conf = res.get("confidence")

            if visual_selected:
                item["selected_options"] = [visual_selected]
                item["visual_override"] = True
                item["visual_selected_option"] = visual_selected
                item["visual_page_num"] = visual_page
                item["visual_confidence"] = visual_conf
                item["found"] = True
                item["page_num"] = visual_page or item.get("page_num")
                item["source"] = f"{item.get('source', 'unknown')} + visual_fallback"

        out[field] = item

    return out


def apply_nature_du_sejour_rule(merged_fields: dict, results: list[dict]) -> dict:
    out = {k: dict(v) for k, v in merged_fields.items()}
    field_name = "Nature du séjour"
    if field_name not in out:
        return out

    current = out[field_name]
    if current.get("selected_options"):
        return out

    filled_autres_text = None
    found_page = None

    for page_result in results:
        page_num = page_result["page_num"]
        full_lines = page_result.get("processed_lines", [])

        for i, line in enumerate(full_lines):
            c = compact_norm(line)

            if "siautrespreciser" in c:
                if ":" in line:
                    right = clean_text(line.split(":", 1)[1])
                    if right:
                        filled_autres_text = right
                        found_page = page_num
                        break

                if i + 1 < len(full_lines):
                    nxt = clean_text(full_lines[i + 1])
                    if nxt and "similitaire" not in compact_norm(nxt):
                        filled_autres_text = nxt
                        found_page = page_num
                        break

        if filled_autres_text:
            break

    if filled_autres_text:
        current["selected_options"] = ["Autres"]
        current["found"] = True
        current["nature_du_sejour_inferred_from_autres_preciser"] = filled_autres_text
        current["source"] = f"{current.get('source', 'unknown')} + autres_preciser_rule"
        current["page_num"] = found_page or current.get("page_num")

    out[field_name] = current
    return out


def _looks_like_placeholder(value: str) -> bool:
    if not value:
        return True

    v = clean_text(value)
    vn = compact_norm(v)

    bad_exact = {
        "jjmmaaaa",
        "jmmaaaa",
        "jjmm",
        "jmm",
        "nom",
        "prenom",
        "idpatient",
        "paysdenaissance",
        "paysderesidence",
        "datedenaissance",
        "datedudiagnosticbiologique",
        "datedepart",
        "dateretour",
        "datepremierssymptomes",
        "datedeladerniereprise",
        "siautrespreciser",
        "similitaireunite",
    }
    if vn in bad_exact:
        return True

    bad_substrings = [
        "jj/mm/aaaa",
        "(jj/mm/aaaa)",
        "(j)/mm",
        "valeurs usuelles",
        "annuler la selection",
        "annuler laselection",
        "le symbole",
        "accueil",
        "deconnecter",
        "page 1 sur",
        "page 2 sur",
        "page 3 sur",
    ]
    nv = norm(v)
    return any(x in nv for x in bad_substrings)


def _looks_like_label_line(value: str) -> bool:
    v = clean_text(value)
    vn = norm(v)
    label_like = [
        "nom",
        "prénom",
        "prenom",
        "id patient",
        "date de naissance",
        "date du diagnostic biologique",
        "pays de naissance",
        "pays de résidence",
        "pays de residence",
        "date départ",
        "date depart",
        "date retour",
        "date des premiers symptômes",
        "date des premiers symptomes",
        "date de la consultation actuelle",
        "date de la dernière prise",
        "date de la derniere prise",
        "unité militaire",
        "unite militaire",
        "si autres, préciser",
        "si autres, preciser",
    ]
    return any(x in vn for x in label_like)


def _contains_other_label(text: str, current_field: str) -> bool:
    txt = norm(text)
    other_labels = {
        "Année": ["nom", "prenom", "id patient", "date de naissance"],
        "ID patient": ["nom", "prenom", "année", "annee"],
        "Nom": ["prenom", "id patient", "date de naissance"],
        "Prénom": ["nom", "id patient", "date de naissance"],
        "Date de naissance": ["nom", "prenom", "id patient"],
        "Date du Diagnostic Biologique": ["nom", "prenom", "id patient"],
        "Pays de naissance": ["pays de résidence", "pays de residence"],
        "Pays de résidence": ["pays de naissance"],
        "Date départ": ["date retour"],
        "Date retour": ["date départ", "date depart"],
        "Date premiers symptômes": ["date de la consultation actuelle"],
        "Date consultation avant": ["date des premiers symptomes", "date des premiers symptômes"],
        "Unité militaire": ["si autres", "autres, preciser", "autres, préciser"],
        "Autres, préciser": ["si militaire", "unité militaire", "unite militaire"],
        "Date de la dernière prise": ["si pas de date"],
    }
    return any(bad in txt for bad in other_labels.get(current_field, []))


def _extract_first_date_from_text(text: str):
    txt = clean_text(text)
    m = re.search(r"\b(\d{2})[^\d]{0,3}(\d{2})[^\d]{0,3}(\d{4})\b", txt)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        try:
            if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                return f"{dd}/{mm}/{yyyy}"
        except Exception:
            pass
    return None


def _extract_date_near_anchor(
    lines: list[str],
    anchors: list[str],
    back_look: int = 0,
    forward_look: int = 8,
):
    stop_labels = [
        "date retour",
        "date depart",
        "date départ",
        "date des premiers symptomes",
        "date des premiers symptômes",
        "date de la consultation actuelle",
        "date du diagnostic biologique",
        "date de naissance",
        "date de la derniere prise",
        "date de la dernière prise",
    ]

    def is_label_line(txt: str) -> bool:
        nt = norm(txt)
        return any(lbl in nt for lbl in stop_labels)

    for i, line in enumerate(lines):
        raw = clean_text(line)
        raw_norm = compact_norm(raw)

        matched_anchor = None
        for anchor in anchors:
            anchor_norm = compact_norm(anchor)
            if anchor_norm and anchor_norm in raw_norm:
                matched_anchor = anchor
                break

        if not matched_anchor:
            continue

        local_lines = []
        start = max(0, i - back_look)
        end = min(len(lines), i + forward_look + 1)

        for j in range(start, end):
            txt = clean_text(lines[j])
            if txt:
                local_lines.append((j, txt))

        for j, cand in local_lines:
            if _looks_like_placeholder(cand):
                continue
            if j != i and is_label_line(cand):
                continue

            dt = _extract_first_date_from_text(cand)
            if dt:
                return dt

        token_pool = []
        for j, cand in local_lines:
            if _looks_like_placeholder(cand):
                continue
            if j != i and is_label_line(cand):
                continue

            nums = re.findall(r"\b\d{1,4}\b", cand)
            if nums:
                token_pool.append((j, nums))

        flat_tokens = []
        for _, nums in token_pool:
            flat_tokens.extend(nums)

        for k in range(len(flat_tokens) - 2):
            a, b, c = flat_tokens[k], flat_tokens[k + 1], flat_tokens[k + 2]
            if len(a) <= 2 and len(b) <= 2 and len(c) == 4:
                dd = a.zfill(2)
                mm = b.zfill(2)
                yyyy = c
                try:
                    if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12 and 1900 <= int(yyyy) <= 2100:
                        return f"{dd}/{mm}/{yyyy}"
                except Exception:
                    pass

    return None


def _extract_page1_travel_dates(lines: list[str]) -> dict:
    out = {"Date retour": None, "Date départ": None}

    def valid_date(dd, mm, yyyy):
        try:
            d = int(dd)
            m = int(mm)
            y = int(yyyy)
            return 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100
        except Exception:
            return False

    def extract_date_from_window(window_lines):
        # Case 1: direct full date on one line
        for txt in window_lines:
            txt = clean_text(txt)
            m = re.search(r"\b(\d{2})[^\d]{0,3}(\d{2})[^\d]{0,3}(\d{4})\b", txt)
            if m:
                dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
                if valid_date(dd, mm, yyyy):
                    return f"{dd}/{mm}/{yyyy}"

        # Case 2: collect numeric tokens after the anchor
        nums = []
        for txt in window_lines:
            txt = clean_text(txt)
            nums.extend(re.findall(r"\b\d{1,4}\b", txt))

        for i in range(len(nums) - 2):
            a, b, c = nums[i], nums[i + 1], nums[i + 2]
            if len(a) <= 2 and len(b) <= 2 and len(c) == 4:
                dd = a.zfill(2)
                mm = b.zfill(2)
                yyyy = c
                if valid_date(dd, mm, yyyy):
                    return f"{dd}/{mm}/{yyyy}"

        return None

    for i, line in enumerate(lines):
        txt = clean_text(line)
        n = norm(txt)

        # -------------------------
        # Date départ
        # -------------------------
        if "date départ" in n or "date depart" in n:
            window = [txt]

            # right side of same line
            if ":" in txt:
                right = clean_text(txt.split(":", 1)[1])
                if right:
                    window.append(right)

            # next few lines only
            for j in range(i + 1, min(len(lines), i + 5)):
                nxt = clean_text(lines[j])
                if not nxt:
                    continue
                nn = norm(nxt)
                if "date retour" in nn:
                    break
                window.append(nxt)

            candidate = extract_date_from_window(window)
            if candidate:
                out["Date départ"] = candidate

        # -------------------------
        # Date retour
        # -------------------------
        if "date retour" in n:
            window = [txt]

            if ":" in txt:
                right = clean_text(txt.split(":", 1)[1])
                if right:
                    window.append(right)

            for j in range(i + 1, min(len(lines), i + 5)):
                nxt = clean_text(lines[j])
                if not nxt:
                    continue
                nn = norm(nxt)
                if "durée du séjour" in nn or "duree du sejour" in nn:
                    break
                window.append(nxt)

            candidate = extract_date_from_window(window)
            if candidate:
                out["Date retour"] = candidate

    return out

def _extract_ethnicite_from_page1(lines: list[str]):
    for line in lines:
        txt = clean_text(line)
        n = norm(txt)

        if "ethnicité:" in n or "ethnicite:" in n:
            if ":" in txt:
                right = clean_text(txt.split(":", 1)[1])
                if right:
                    right_n = norm(right)

                    if "africain vivant en france" in right_n:
                        return "Africain vivant en France"
                    if "africain vivant en afrique" in right_n:
                        return "Africain vivant en Afrique"
                    if "africain" in right_n:
                        return "Africain"
                    if "caucasien" in right_n:
                        return "Caucasien"
                    if "asiatique" in right_n:
                        return "Asiatique"
                    if "autre" in right_n:
                        return "Autre"
                    if "nsp" in right_n:
                        return "NSP"
    return None


def _extract_year_near_anchor(lines, anchors, max_lookahead: int = 3):
    for i, line in enumerate(lines):
        raw = clean_text(line)
        raw_norm = compact_norm(raw)

        for anchor in anchors:
            anchor_norm = compact_norm(anchor)
            if not anchor_norm or anchor_norm not in raw_norm:
                continue

            candidates = [raw]
            for j in range(1, max_lookahead + 1):
                if i + j < len(lines):
                    nxt = clean_text(lines[i + j])
                    if nxt:
                        candidates.append(nxt)

            for cand in candidates:
                if _looks_like_placeholder(cand):
                    continue
                m = re.search(r"\b(19|20)\d{2}\b", cand)
                if m:
                    return m.group(0)
    return None


def _extract_id_patient(lines, anchors, max_lookahead: int = 4):
    for i, line in enumerate(lines):
        raw = clean_text(line)
        raw_norm = compact_norm(raw)

        for anchor in anchors:
            anchor_norm = compact_norm(anchor)
            if not anchor_norm or anchor_norm not in raw_norm:
                continue

            candidates = []

            if ":" in raw:
                left, right = raw.split(":", 1)
                if anchor_norm in compact_norm(left):
                    candidates.append(clean_text(right))

            for j in range(1, max_lookahead + 1):
                if i + j < len(lines):
                    nxt = clean_text(lines[i + j])
                    if nxt:
                        candidates.append(nxt)

            for cand in candidates:
                if _looks_like_placeholder(cand):
                    continue

                m = re.search(r"\brdb\d+\b", cand, flags=re.IGNORECASE)
                if m:
                    return m.group(0)

                m = re.search(r"\b[a-zA-Z]{1,4}\d+[a-zA-Z0-9]*\b", cand)
                if m:
                    return m.group(0)

                if re.fullmatch(r"[A-Za-z0-9_./()-]{3,}", cand):
                    return cand

    return None


def _extract_inline_value_only(lines, anchors):
    for line in lines:
        raw = clean_text(line)
        raw_norm = compact_norm(raw)

        for anchor in anchors:
            anchor_norm = compact_norm(anchor)
            if not anchor_norm or anchor_norm not in raw_norm:
                continue

            if ":" in raw:
                left, right = raw.split(":", 1)
                if anchor_norm in compact_norm(left):
                    right = clean_text(right)
                    if right and not _looks_like_placeholder(right) and not _looks_like_label_line(right):
                        return right

    return None


def _extract_same_line_text_after_anchor(lines, anchors, field_name: str, max_lookahead: int = 3):
    for i, line in enumerate(lines):
        raw = clean_text(line)
        raw_norm = compact_norm(raw)

        for anchor in anchors:
            anchor_norm = compact_norm(anchor)
            if not anchor_norm or anchor_norm not in raw_norm:
                continue

            candidates = []

            if ":" in raw:
                left, right = raw.split(":", 1)
                if anchor_norm in compact_norm(left):
                    right = clean_text(right)
                    if right:
                        candidates.append(right)

            for j in range(1, max_lookahead + 1):
                if i + j < len(lines):
                    nxt = clean_text(lines[i + j])
                    if nxt:
                        candidates.append(nxt)

            for cand in candidates:
                if _looks_like_placeholder(cand):
                    continue
                if _contains_other_label(cand, field_name):
                    continue
                if _looks_like_label_line(cand):
                    continue

                if field_name in {"Nom", "Prénom"}:
                    if re.fullmatch(r"[A-Za-zÀ-ÿ' -]{2,}", cand):
                        return cand
                elif field_name in {"Pays de naissance", "Pays de résidence", "Autres, préciser"}:
                    if len(cand) >= 2:
                        return cand
                elif field_name == "Unité militaire":
                    if "si autres" in norm(cand):
                        continue
                    if len(cand) >= 2:
                        return cand
                else:
                    return cand

    return None


def extract_simple_text_fields(results: list[dict]) -> dict:
    field_specs = {
        "Année": {"anchors": ["Année", "Annee"], "kind": "year", "preferred_pages": [1, 3]},
        "ID patient": {"anchors": ["ID patient", "Id"], "kind": "id", "preferred_pages": [1, 3]},
        "Nom": {"anchors": ["Nom"], "kind": "inline_text_only", "preferred_pages": [3, 1]},
        "Prénom": {"anchors": ["Prénom", "Prenom"], "kind": "inline_text_only", "preferred_pages": [3, 1]},
        "Date de naissance": {"anchors": ["Date de naissance"], "kind": "date", "preferred_pages": [1, 3]},
        "Date du Diagnostic Biologique": {
            "anchors": ["Date du Diagnostic Biologique"],
            "kind": "date",
            "preferred_pages": [3],
        },
        "Pays de naissance": {"anchors": ["Pays de naissance"], "kind": "text", "preferred_pages": [1]},
        "Pays de résidence": {
            "anchors": ["Pays de résidence", "Pays de residence"],
            "kind": "text",
            "preferred_pages": [1],
        },
        "Date départ": {"anchors": ["Date départ", "Date depart"], "kind": "date", "preferred_pages": [1]},
        "Date retour": {"anchors": ["Date retour"], "kind": "date", "preferred_pages": [1]},
        "Date premiers symptômes": {
            "anchors": ["Date des Premiers Symptômes", "Date des Premiers Symptomes"],
            "kind": "date",
            "preferred_pages": [2],
        },
        "Date consultation avant": {
            "anchors": ["Date de la consultation actuelle"],
            "kind": "date",
            "preferred_pages": [2],
        },
        "Unité militaire": {
            "anchors": ["Si militaire, unité", "Si militaire, unite"],
            "kind": "inline_text_only",
            "preferred_pages": [1],
        },
        "Autres, préciser": {
            "anchors": ["Si autres, préciser", "Si autres, preciser"],
            "kind": "text",
            "preferred_pages": [1],
        },
        "Date de la dernière prise": {
            "anchors": [
                "Date de la dernière prise",
                "Date de la demière prise",
                "Date de la derniere prise",
            ],
            "kind": "date",
            "preferred_pages": [4],
        },
    }

    page_map = {r["page_num"]: r.get("processed_lines", []) for r in results}
    page1_lines = page_map.get(1, [])
    page1_travel_dates = _extract_page1_travel_dates(page1_lines) if page1_lines else {}

    extracted = {
        field_name: {"value": None, "page_num": None}
        for field_name in field_specs
    }

    for field_name, spec in field_specs.items():
        search_pages = list(spec["preferred_pages"])
        for p in sorted(page_map.keys()):
            if p not in search_pages:
                search_pages.append(p)

        value = None
        value_page = None

        for page_num in search_pages:
            lines = page_map.get(page_num, [])
            if not lines:
                continue

            kind = spec["kind"]
            anchors = spec["anchors"]

            if field_name == "Date retour" and page_num == 1:
                candidate = page1_travel_dates.get("Date retour")
            elif field_name == "Date départ" and page_num == 1:
                candidate = page1_travel_dates.get("Date départ")
            elif kind == "date":
                candidate = _extract_date_near_anchor(lines, anchors)
            elif kind == "year":
                candidate = _extract_year_near_anchor(lines, anchors)
            elif kind == "id":
                candidate = _extract_id_patient(lines, anchors)
            elif kind == "inline_text_only":
                candidate = _extract_inline_value_only(lines, anchors)
            else:
                candidate = _extract_same_line_text_after_anchor(lines, anchors, field_name)

            if candidate and not _looks_like_placeholder(candidate):
                value = candidate
                value_page = page_num
                break

        extracted[field_name] = {"value": value, "page_num": value_page}

    return extracted

def run_specs(lines: list[str], specs: list[dict], source_name: str):
    sections = []
    field_results = []

    for spec in specs:
        sec = slice_section(
            lines=lines,
            start_anchors=spec["start_anchors"],
            end_anchors=spec.get("end_anchors"),
            max_lines=spec.get("max_lines", 20),
        )

        if sec is None:
            sections.append({"field": spec["field"], "found": False, "source": source_name})
            field_results.append(
                {
                    "field": spec["field"],
                    "found": False,
                    "source": source_name,
                    "selected_options": [],
                    "options": [],
                }
            )
            continue

        sections.append(
            {
                "field": spec["field"],
                "found": True,
                "source": source_name,
                "anchor_line": sec["anchor_line"],
                "start_idx": sec["start_idx"],
                "end_idx": sec["end_idx"],
                "lines": sec["lines"],
            }
        )

        option_results = []
        for canonical, variants in spec["options"]:
            option_results.append(parse_option_from_lines(sec["lines"], canonical, variants))

        option_results = postprocess_single_choice(option_results)
        option_results = apply_elimination_heuristic(option_results, spec.get("single_choice", False))
        selected_options = [x["option"] for x in option_results if x.get("selected")]

        field_results.append(
            {
                "field": spec["field"],
                "found": True,
                "source": source_name,
                "selected_options": selected_options,
                "options": option_results,
            }
        )

    return sections, field_results


def safe_custom_block(func, lines: list[str], block_name: str):
    try:
        result = func(lines)
        return {"block": block_name, "success": True, "result": result}
    except Exception as e:
        return {"block": block_name, "success": False, "error": str(e)}


def run_all_specs_on_page(lines: list[str]):
    all_sections = []
    all_field_results = []

    spec_groups = [
        ("page1_specs", page1_specs()),
        ("page2_specs", page2_specs()),
        ("page3_specs", page3_specs()),
        ("page4_specs", page4_specs()),
    ]

    for source_name, specs in spec_groups:
        sections, field_results = run_specs(lines, specs, source_name)
        all_sections.extend(sections)
        all_field_results.extend(field_results)

    custom_blocks = [
        safe_custom_block(parse_page2_clinical_block, lines, "parse_page2_clinical_block"),
        safe_custom_block(parse_examens_parasitologiques_block, lines, "parse_examens_parasitologiques_block"),
        safe_custom_block(parse_protection_personnelle_block, lines, "parse_protection_personnelle_block"),
        safe_custom_block(parse_chimioprophylaxie_block, lines, "parse_chimioprophylaxie_block"),
        safe_custom_block(parse_bandelettes_block, lines, "parse_bandelettes_block"),
        safe_custom_block(parse_page3_lab_values_block, lines, "parse_page3_lab_values_block"),
        safe_custom_block(parse_controle_parasitologique, lines, "parse_controle_parasitologique"),
        safe_custom_block(parse_page4_treatment_block, lines, "parse_page4_treatment_block"),
    ]

    return {
        "sections": all_sections,
        "field_results": all_field_results,
        "custom_blocks": custom_blocks,
    }


def merge_across_pages(results: list[dict]) -> dict:
    merged: dict[str, dict] = {}

    for page_result in results:
        page_num = page_result["page_num"]
        for fr in page_result["all_page_parsers_output"]["field_results"]:
            field = fr["field"]
            if field not in merged:
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num
                continue

            existing = merged[field]

            if fr.get("selected_options") and not existing.get("selected_options"):
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num
            elif fr.get("found") and not existing.get("found"):
                merged[field] = dict(fr)
                merged[field]["page_num"] = page_num

    return merged


def custom_block_has_meaningful_result(res: dict) -> bool:
    field = res.get("field")

    if field == "Examens parasitologiques":
        if res.get("frottis_mince", {}).get("status"):
            return True
        if res.get("goutte_epaisse", {}).get("status"):
            return True
        if res.get("bandelettes", {}).get("status"):
            return True
        if res.get("autres_techniques", {}).get("found_options"):
            return True
        return False

    if field == "Valeurs biologiques":
        values = [
            res.get("Hemoglobine (g/l)"),
            res.get("GR (tera/l)"),
            res.get("GB (giga/l)"),
            res.get("Plaquettes (giga/l)"),
        ]
        return sum(1 for v in values if v not in (None, "", "None")) >= 3

    if field == "Bandelettes":
        return bool(res.get("found")) and (
            bool(res.get("status"))
            or bool(res.get("selected_results"))
            or bool(res.get("all_results_found"))
        )

    if field == "Protection Personnelle Anti-Moustiques":
        if res.get("selected"):
            return True
        for d in res.get("details", []):
            if d.get("selection") or d.get("options_found"):
                return True
        return False

    if field == "Chimioprophylaxie utilisée":
        return bool(res.get("found")) and (
            bool(res.get("selected")) or bool(res.get("table_results"))
        )

    if field == "Traitement et hospitalisation":
        return any(
            [
                res.get("prise_en_charge"),
                res.get("date_premiere_prise_structure"),
                res.get("nombre_de_jours_hospitalisation"),
                res.get("dont_reanimation_si"),
                res.get("transfert_autre_hopital"),
                res.get("poids_kg"),
                bool(res.get("traitement_antipalustre")),
                res.get("dose_totale_mg_j"),
                res.get("duree_jours"),
                res.get("commentaires"),
            ]
        )

    if field == "Contrôle parasitologique P falciparum":
        if res.get("control_overall"):
            return True
        for row in res.get("rows", []):
            if (
                row.get("fait") is not None
                or row.get("temperature")
                or row.get("parasitologie")
                or row.get("densite_parasitaire")
            ):
                return True
        if res.get("commentaires_remarques"):
            return True
        if res.get("perdu_de_vue") is not None:
            return True
        return False

    if field == "Contexte clinique page 2":
        return any(
            [
                res.get("date_consultation_actuelle"),
                res.get("etat_clinique"),
                res.get("antecedents_paludisme_3m"),
                res.get("femme_enceinte_ou_parturiente"),
                res.get("immunodepression_connue"),
                res.get("paludismes_autochtones"),
            ]
        )

    return bool(res.get("found"))


def merge_controle_blocks(text_res: dict | None, visual_res: dict | None) -> dict | None:
    if not text_res and not visual_res:
        return None
    if not text_res:
        return visual_res
    if not visual_res:
        return text_res

    merged = {
        "field": "Contrôle parasitologique P falciparum",
        "found": bool(text_res.get("found") or visual_res.get("found")),
        "control_overall": visual_res.get("control_overall") or text_res.get("control_overall"),
        "rows": [],
        "merge_source": "text+visual",
        "commentaires_remarques": text_res.get("commentaires_remarques") or visual_res.get("commentaires_remarques"),
        "perdu_de_vue": (
            text_res.get("perdu_de_vue")
            if text_res.get("perdu_de_vue") is not None
            else visual_res.get("perdu_de_vue")
        ),
    }

    text_rows = {r.get("row"): r for r in text_res.get("rows", [])}
    visual_rows = {r.get("row"): r for r in visual_res.get("rows", [])}
    ordered_rows = ["J3 ou J4", "J7 +/-1", "J28 +/-2", "Autre"]

    for row_name in ordered_rows:
        tr = text_rows.get(row_name, {})
        vr = visual_rows.get(row_name, {})

        merged_row = {
            "row": row_name,
            "fait": vr.get("fait") or tr.get("fait"),
            "temperature": vr.get("temperature") or tr.get("temperature"),
            "parasitologie": vr.get("parasitologie") or tr.get("parasitologie") or [],
            "densite_parasitaire": vr.get("densite_parasitaire") or tr.get("densite_parasitaire") or [],
        }

        if not merged_row["parasitologie"]:
            merged_row["parasitologie"] = []
        if not merged_row["densite_parasitaire"]:
            merged_row["densite_parasitaire"] = []

        merged["rows"].append(merged_row)

    return merged


def collect_meaningful_custom_blocks(ocr_txt_path: Path, results: list[dict]) -> list[dict]:
    meaningful = []
    controle_text_res = None

    for page_result in results:
        page_num = page_result["page_num"]
        parsed = page_result.get("all_page_parsers_output", {})
        custom_blocks = parsed.get("custom_blocks", [])

        for block in custom_blocks:
            if not block.get("success", False):
                continue

            res = block.get("result", {})
            if not custom_block_has_meaningful_result(res):
                continue

            field_name = res.get("field")

            if field_name == "Contrôle parasitologique P falciparum":
                if controle_text_res is None:
                    controle_text_res = res
                else:
                    controle_text_res = merge_controle_blocks(controle_text_res, res)
                continue

            meaningful.append({"page_num": page_num, "result": res})

    controle_visual_res = run_controle_parasito_visual_fallback(ocr_txt_path)
    merged_controle = merge_controle_blocks(controle_text_res, controle_visual_res)
    if merged_controle:
        meaningful.append({"page_num": 5, "result": merged_controle})

    return meaningful


def enrich_merged_fields_from_custom_blocks(merged_fields: dict, meaningful_blocks: list[dict]) -> dict:
    out = {k: dict(v) for k, v in merged_fields.items()}

    for item in meaningful_blocks:
        res = item.get("result", {})
        field = res.get("field")

        if field == "Bandelettes":
            selected_results = res.get("selected_results", [])
            if selected_results and (not out.get("Bandelettes", {}).get("selected_options")):
                out.setdefault("Bandelettes", {"field": "Bandelettes", "found": True, "selected_options": []})
                out["Bandelettes"]["selected_options"] = selected_results
                out["Bandelettes"]["found"] = True
                out["Bandelettes"]["source"] = "custom_block:bandelettes"

        elif field == "Protection Personnelle Anti-Moustiques":
            sel = res.get("selected")
            if sel and (not out.get("Protection Personnelle Anti-Moustiques", {}).get("selected_options")):
                out.setdefault(
                    "Protection Personnelle Anti-Moustiques",
                    {"field": "Protection Personnelle Anti-Moustiques", "found": True, "selected_options": []},
                )
                out["Protection Personnelle Anti-Moustiques"]["selected_options"] = [sel]
                out["Protection Personnelle Anti-Moustiques"]["found"] = True
                out["Protection Personnelle Anti-Moustiques"]["source"] = "custom_block:protection"

            for detail in res.get("details", []):
                item_name = detail.get("item")
                selection = detail.get("selection")
                if not selection:
                    continue

                if item_name == "Répulsifs cutanés":
                    out.setdefault("Répulsifs cutanés", {"field": "Répulsifs cutanés", "found": True, "selected_options": []})
                    out["Répulsifs cutanés"]["selected_options"] = [selection]
                    out["Répulsifs cutanés"]["found"] = True
                    out["Répulsifs cutanés"]["source"] = "custom_block:protection_detail"

                elif item_name == "Moustiquaires de lit":
                    out.setdefault("Moustiquaires de lit", {"field": "Moustiquaires de lit", "found": True, "selected_options": []})
                    out["Moustiquaires de lit"]["selected_options"] = [selection]
                    out["Moustiquaires de lit"]["found"] = True
                    out["Moustiquaires de lit"]["source"] = "custom_block:protection_detail"

        elif field == "Chimioprophylaxie utilisée":
            sel = res.get("selected")
            if sel and (not out.get("Chimioprophylaxie utilisée", {}).get("selected_options")):
                out.setdefault("Chimioprophylaxie utilisée", {"field": "Chimioprophylaxie utilisée", "found": True, "selected_options": []})
                out["Chimioprophylaxie utilisée"]["selected_options"] = [sel]
                out["Chimioprophylaxie utilisée"]["found"] = True
                out["Chimioprophylaxie utilisée"]["source"] = "custom_block:chimioprophylaxie"

        elif field == "Traitement et hospitalisation":
            pe = res.get("prise_en_charge")
            if pe:
                values = [x.strip() for x in pe.split(",") if x.strip()]
                if values:
                    out.setdefault("Prise en charge", {"field": "Prise en charge", "found": True, "selected_options": []})
                    out["Prise en charge"]["selected_options"] = values
                    out["Prise en charge"]["found"] = True
                    out["Prise en charge"]["source"] = "custom_block:treatment"

        elif field == "Contrôle parasitologique P falciparum":
            overall = res.get("control_overall")
            if overall and (not out.get("Contrôle parasitologique P falciparum", {}).get("selected_options")):
                out.setdefault(
                    "Contrôle parasitologique P falciparum",
                    {"field": "Contrôle parasitologique P falciparum", "found": True, "selected_options": []},
                )
                out["Contrôle parasitologique P falciparum"]["selected_options"] = [overall]
                out["Contrôle parasitologique P falciparum"]["found"] = True
                out["Contrôle parasitologique P falciparum"]["source"] = "custom_block:controle"

    return out


def enrich_custom_blocks_for_display(meaningful_blocks: list[dict], merged_fields: dict, simple_text_fields: dict):
    out = []

    for item in meaningful_blocks:
        page_num = item.get("page_num")
        res = dict(item.get("result", {}))
        field = res.get("field")

        if field == "Contexte clinique page 2":
            def merged_value(name: str):
                fr = merged_fields.get(name, {})
                sel = fr.get("selected_options", [])
                if sel:
                    return ", ".join(sel)
                return None

            def simple_value(name: str):
                x = simple_text_fields.get(name, {})
                return x.get("value")

            res["date_consultation_actuelle"] = (
                res.get("date_consultation_actuelle")
                or simple_value("Date consultation avant")
            )
            res["etat_clinique"] = (
                res.get("etat_clinique")
                or merged_value("Etat clinique au moment du diagnostic")
            )
            res["antecedents_paludisme_3m"] = (
                res.get("antecedents_paludisme_3m")
                or merged_value("Antécédents de paludisme dans les 3 derniers mois")
            )
            res["femme_enceinte_ou_parturiente"] = (
                res.get("femme_enceinte_ou_parturiente")
                or merged_value("Femme enceinte ou parturiente")
            )
            res["immunodepression_connue"] = (
                res.get("immunodepression_connue")
                or merged_value("Immunodépression connue")
            )
            res["paludismes_autochtones"] = (
                res.get("paludismes_autochtones")
                or merged_value("Paludismes autochtones")
            )

        out.append({"page_num": page_num, "result": res})

    return out


def build_full_document_txt(
    ocr_txt_path: Path,
    pages: list[dict],
    merged_fields: dict,
    results: list[dict],
    simple_text_fields: dict,
) -> str:
    lines = []
    lines.append(f"OCR TXT PATH: {ocr_txt_path}")
    lines.append(f"Physical pages found: {len(pages)}")
    lines.append("")

    lines.append("=== FINAL STRUCTURED ANSWERS ===")
    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)
        value = ", ".join(sel) if sel else ("None" if found else "None")
        lines.append(f"{field}: {value}")

    lines.append("")
    lines.append("=== SIMPLE OCR TEXT FIELDS ===")
    any_simple = False
    for field, item in simple_text_fields.items():
        value = item.get("value")
        page_num = item.get("page_num")
        if value:
            any_simple = True
            lines.append(f"[{page_num}] {field}: {value}")
    if not any_simple:
        lines.append("None")

    lines.append("")
    lines.append("=== CUSTOM BLOCKS ===")

    meaningful_blocks = collect_meaningful_custom_blocks(ocr_txt_path, results)
    meaningful_blocks = enrich_custom_blocks_for_display(
        meaningful_blocks,
        merged_fields,
        simple_text_fields,
    )
    if not meaningful_blocks:
        lines.append("None")
        return "\n".join(lines)

    for item in meaningful_blocks:
        res = item["result"]
        field = res.get("field", "unknown_block")

        if field == "Contexte clinique page 2":
            lines.append(f"{field}:")
            lines.append(f"  Date de la consultation actuelle: {res.get('date_consultation_actuelle') or 'None'}")
            lines.append(f"  Etat clinique au moment du diagnostic: {res.get('etat_clinique') or 'None'}")
            lines.append(f"  Antécédents de paludisme dans les 3 derniers mois: {res.get('antecedents_paludisme_3m') or 'None'}")
            lines.append(f"  Femme enceinte ou parturiente: {res.get('femme_enceinte_ou_parturiente') or 'None'}")
            lines.append(f"  Immunodépression connue: {res.get('immunodepression_connue') or 'None'}")
            lines.append(f"  Paludismes autochtones: {res.get('paludismes_autochtones') or 'None'}")
            continue

        if field == "Examens parasitologiques":
            fm = res.get("frottis_mince", {})
            ge = res.get("goutte_epaisse", {})
            bd = res.get("bandelettes", {})
            at = res.get("autres_techniques", {})

            lines.append(f"{field}:")
            lines.append(f"  Frottis mince: {fm.get('status') if fm.get('status') else 'None'}")
            lines.append(f"    %: {', '.join(fm.get('percent_values', [])) if fm.get('percent_values') else 'None'}")
            lines.append(f"    Densité: {', '.join(fm.get('count_values', [])) if fm.get('count_values') else 'None'}")

            lines.append(f"  Goutte épaisse: {ge.get('status') if ge.get('status') else 'None'}")
            lines.append(f"    /1000 GB: {', '.join(ge.get('per_1000_gb_values', [])) if ge.get('per_1000_gb_values') else 'None'}")
            lines.append(f"    Densité: {', '.join(ge.get('count_values', [])) if ge.get('count_values') else 'None'}")

            lines.append(f"  Bandelettes: {bd.get('status') if bd.get('status') else 'None'}")

            selected_at = at.get("selected_options", [])
            lines.append(f"  Autres techniques: {', '.join(selected_at) if selected_at else 'None'}")
            continue

        if field == "Valeurs biologiques":
            lines.append(f"{field}:")
            lines.append(f"  Hemoglobine (g/l): {res.get('Hemoglobine (g/l)') or 'None'}")
            lines.append(f"  GR (tera/l): {res.get('GR (tera/l)') or 'None'}")
            lines.append(f"  GB (giga/l): {res.get('GB (giga/l)') or 'None'}")
            lines.append(f"  Plaquettes (giga/l): {res.get('Plaquettes (giga/l)') or 'None'}")
            continue

        if field == "Protection Personnelle Anti-Moustiques":
            selected = res.get("selected")
            lines.append(f"{field}: {selected if selected else 'None'}")
            for d in res.get("details", []):
                item_name = d.get("item")
                selection = d.get("selection")
                if selection:
                    lines.append(f"  {item_name}: {selection}")
            continue

        if field == "Chimioprophylaxie utilisée":
            selected = res.get("selected")
            lines.append(f"{field}: {selected if selected else 'None'}")
            table_results = res.get("table_results", [])
            for row in table_results:
                med = row.get("medicine")
                sel = row.get("selection")
                lines.append(f"  {med}: {sel if sel else 'None'}")
            continue

        if field == "Bandelettes":
            status = res.get("status")
            selected_results = res.get("selected_results", [])
            value_parts = []
            if status:
                value_parts.append(status)
            if selected_results:
                value_parts.append(", ".join(selected_results))
            lines.append(f"{field}: {' | '.join(value_parts) if value_parts else 'None'}")
            continue

        if field == "Traitement et hospitalisation":
            lines.append(f"{field}:")
            lines.append(f"  Prise en charge: {res.get('prise_en_charge') or 'None'}")
            lines.append(f"  Date de la première prise dans votre structure: {res.get('date_premiere_prise_structure') or 'None'}")
            lines.append(f"  Nombre de jours d’hospitalisation: {res.get('nombre_de_jours_hospitalisation') or 'None'}")
            lines.append(f"  dont réanimation/SI: {res.get('dont_reanimation_si') or 'None'}")
            lines.append(f"  Transfert autre hôpital: {res.get('transfert_autre_hopital') or 'None'}")
            lines.append(f"  Poids (kg): {res.get('poids_kg') or 'None'}")
            meds = res.get("traitement_antipalustre", [])
            lines.append(f"  Traitement anti-palustre: {', '.join(meds) if meds else 'None'}")
            lines.append(f"  Traitement débuté le: {res.get('traitement_debute_le') or 'None'}")
            lines.append(f"  Dose totale (mg/j): {res.get('dose_totale_mg_j') or 'None'}")
            lines.append(f"  Durée en jours: {res.get('duree_jours') or 'None'}")
            lines.append(f"  Commentaires: {res.get('commentaires') or 'None'}")
            continue

        if field == "Contrôle parasitologique P falciparum":
            overall = res.get("control_overall")
            lines.append(f"{field}: {overall if overall else 'None'}")
            for row in res.get("rows", []):
                row_name = row.get("row")
                fait = row.get("fait")
                temp = row.get("temperature")
                paras = ", ".join(row.get("parasitologie", [])) if row.get("parasitologie") else "None"
                dens = ", ".join(row.get("densite_parasitaire", [])) if row.get("densite_parasitaire") else "None"

                lines.append(f"  {row_name}: {fait if fait else 'None'}")
                lines.append(f"    Température: {temp if temp else 'None'}")
                lines.append(f"    Parasitologie: {paras}")
                lines.append(f"    Densité parasitaire: {dens}")

            lines.append(
                f"  Commentaires & Remarques: {res.get('commentaires_remarques') or 'None'}"
            )
            lines.append(
                f"  Perdu de vue: {res.get('perdu_de_vue') if res.get('perdu_de_vue') is not None else 'None'}"
            )
            continue

        lines.append(f"{field}: {json.dumps(res, ensure_ascii=False)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Apply all page specs to every page in OCR txt")
    parser.add_argument("ocr_txt_path", help="Path to OCR txt file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_lines = load_ocr_txt(ocr_txt_path)
    pages = extract_all_pages(raw_lines)
    results = []

    for page in pages:
        processed_lines = preprocess_page_lines(page["page_num"], page["lines"])
        parsed = run_all_specs_on_page(processed_lines)
        results.append(
            {
                "page_num": page["page_num"],
                "lines_count": len(processed_lines),
                "processed_lines": processed_lines,
                "processed_lines_preview": processed_lines[:120],
                "all_page_parsers_output": parsed,
            }
        )

    merged_fields = merge_across_pages(results)
    page_map = {r["page_num"]: r.get("processed_lines", []) for r in results}
    page1_lines = page_map.get(1, [])
    ethnicite_direct = _extract_ethnicite_from_page1(page1_lines)

    if ethnicite_direct:
        merged_fields.setdefault("Ethnicité", {"field": "Ethnicité"})
        merged_fields["Ethnicité"]["selected_options"] = [ethnicite_direct]
        merged_fields["Ethnicité"]["found"] = True
        merged_fields["Ethnicité"]["page_num"] = 1
        merged_fields["Ethnicité"]["source"] = "direct_ethnicite_line"
    merged_fields = apply_visual_fallbacks(ocr_txt_path, merged_fields)
    merged_fields = apply_nature_du_sejour_rule(merged_fields, results)

    simple_text_fields = extract_simple_text_fields(results)

    meaningful_blocks = collect_meaningful_custom_blocks(ocr_txt_path, results)
    meaningful_blocks = enrich_custom_blocks_for_display(
        meaningful_blocks,
        merged_fields,
        simple_text_fields,
    )
    merged_fields = enrich_merged_fields_from_custom_blocks(merged_fields, meaningful_blocks)

    final_result = {
        "ocr_txt_path": str(ocr_txt_path),
        "pages_detected": [p["page_num"] for p in pages],
        "pages_count": len(pages),
        "merged_field_results": merged_fields,
        "simple_text_fields": simple_text_fields,
        "results": results,
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_all_pages_all_specs.json"
    out_json.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = []
    summary_lines.append(f"Pages detected: {len(pages)}")
    summary_lines.append(f"Merged fields:  {len(merged_fields)}")
    summary_lines.append(f"Saved JSON: {out_json}")
    summary_lines.append("")
    summary_lines.append("=== MERGED SELECTIONS SUMMARY ===")

    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)
        page = fr.get("page_num", "?")
        status = ", ".join(sel) if sel else ("(found, no selection)" if found else "(not found)")
        summary_lines.append(f"  [{page}] {field}: {status}")

    out_txt = out_dir / f"{ocr_txt_path.stem}_all_pages_all_specs_summary.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved summary TXT: {out_txt}")
    print(f"Pages detected: {len(pages)}")
    print(f"Merged fields:  {len(merged_fields)}")
    print(f"Saved JSON: {out_json}")

    print("\n=== MERGED SELECTIONS SUMMARY ===")
    for field, fr in merged_fields.items():
        sel = fr.get("selected_options", [])
        found = fr.get("found", False)
        page = fr.get("page_num", "?")
        status = ", ".join(sel) if sel else ("(found, no selection)" if found else "(not found)")
        print(f"  [{page}] {field}: {status}")

    full_txt = build_full_document_txt(
        ocr_txt_path,
        pages,
        merged_fields,
        results,
        simple_text_fields,
    )
    out_full_txt = out_dir / f"{ocr_txt_path.stem}_full_final_output.txt"
    out_full_txt.write_text(full_txt, encoding="utf-8")
    print(f"Saved full final TXT: {out_full_txt}")


if __name__ == "__main__":
    main()
from pathlib import Path
import argparse
import json
import re

import cv2
import numpy as np

from debug_visual_utils import ensure_dir, draw_many_boxes, save_image


def clean_text(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_accents_basic(s: str) -> str:
    repl = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "ù": "u", "û": "u",
        "ï": "i", "î": "i",
        "ô": "o", "ö": "o",
        "ç": "c",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "À": "A", "Â": "A",
        "Ù": "U", "Û": "U",
        "Ï": "I", "Î": "I",
        "Ô": "O", "Ö": "O",
        "Ç": "C",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def norm(s: str) -> str:
    s = clean_text(s)
    s = s.replace("：", ":").replace("’", "'").replace("−", "-")
    s = strip_accents_basic(s).lower()
    return clean_text(s)


def compact_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9><=+/\-]", "", norm(s))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_page_words(data: dict, page_num: int = 1):
    out = []

    def add_word(text, box):
        if not text or box is None:
            return

        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
        elif len(box) == 4 and isinstance(box[0], (list, tuple)):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            return

        txt = clean_text(text)
        if not txt:
            return

        out.append({
            "text": txt,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": (float(x1) + float(x2)) / 2.0,
            "cy": (float(y1) + float(y2)) / 2.0,
            "w": float(x2) - float(x1),
            "h": float(y2) - float(y1),
        })

    if isinstance(data, dict) and "overall_ocr_res" in data:
        ocr = data["overall_ocr_res"]
        if isinstance(ocr, dict) and "rec_texts" in ocr and "dt_polys" in ocr:
            for txt, box in zip(ocr["rec_texts"], ocr["dt_polys"]):
                add_word(txt, box)
            return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    if isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            txt = w.get("text") or w.get("word_text") or w.get("transcription")
            box = w.get("box") or w.get("bbox") or w.get("points")
            add_word(txt, box)
        return sorted(out, key=lambda w: (w["y1"], w["x1"]))

    return sorted(out, key=lambda w: (w["y1"], w["x1"]))


def crop_box(img_bgr, box):
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img_bgr[y1:y2, x1:x2].copy()


def window_words(words, x1=None, y1=None, x2=None, y2=None):
    out = []
    for w in words:
        if x1 is not None and w["x2"] < x1:
            continue
        if x2 is not None and w["x1"] > x2:
            continue
        if y1 is not None and w["y2"] < y1:
            continue
        if y2 is not None and w["y1"] > y2:
            continue
        out.append(w)
    return out


def text_matches(word_text: str, variants: list[str]) -> bool:
    wt = norm(word_text)
    wc = compact_norm(word_text)

    for v in variants:
        vn = norm(v)
        vc = compact_norm(v)

        if wt == vn or wc == vc:
            return True
        if vc and vc in wc:
            return True

    return False


def circle_marker_box_for_word(word, left_width=42, right_gap=4, y_pad=8):
    return [
        int(word["x1"] - left_width),
        int(word["y1"] - y_pad),
        int(word["x1"] - right_gap),
        int(word["y2"] + y_pad),
    ]


def score_circle_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "state": "unknown",
            "dark_ratio": 0.0,
            "center_dark_ratio": 0.0,
            "ring_dark_ratio": 0.0,
            "component_count": 0,
            "largest_area": 0,
        }

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    h, w = th.shape[:2]
    total = float(h * w) if h > 0 and w > 0 else 1.0
    dark_ratio = float(np.count_nonzero(th)) / total

    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = max(1.0, w / 2.0)
    ry = max(1.0, h / 2.0)

    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    center_mask = dist <= 0.22
    ring_mask = (dist > 0.22) & (dist <= 0.82)

    center_total = max(1, int(np.count_nonzero(center_mask)))
    ring_total = max(1, int(np.count_nonzero(ring_mask)))

    center_dark_ratio = float(np.count_nonzero(th[center_mask])) / center_total
    ring_dark_ratio = float(np.count_nonzero(th[ring_mask])) / ring_total

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 3:
            areas.append(area)

    largest_area = max(areas) if areas else 0
    component_count = len(areas)

    if center_dark_ratio >= 0.10 or dark_ratio >= 0.10:
        state = "selected"
    elif ring_dark_ratio >= 0.025 or dark_ratio >= 0.02 or largest_area >= 8:
        state = "unselected"
    else:
        state = "unknown"

    return {
        "state": state,
        "dark_ratio": round(dark_ratio, 4),
        "center_dark_ratio": round(center_dark_ratio, 4),
        "ring_dark_ratio": round(ring_dark_ratio, 4),
        "component_count": component_count,
        "largest_area": largest_area,
    }


CIRCLE_FIELD_PRESETS = {
    "sexe": {
        "anchor_variants": ["Sexe"],
        "options": [
            {"label": "M", "variants": ["M"]},
            {"label": "F", "variants": ["F"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 650, "y_before": 5, "y_after": 65},
    },
    "ethnicite": {
        "anchor_variants": ["Ethnicité", "Ethnicite"],
        "options": [
            {"label": "Caucasien", "variants": ["Caucasien"]},
            {"label": "Asiatique", "variants": ["Asiatique"]},
            {"label": "Africain", "variants": ["Africain"]},
            {"label": "Africain vivant en France", "variants": ["Africain vivant en France"]},
            {"label": "Africain vivant en Afrique", "variants": ["Africain vivant en Afrique"]},
            {"label": "Autre", "variants": ["Autre"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 950, "y_before": 5, "y_after": 110},
    },
    "autres_pays_endemie": {
        "anchor_variants": ["Autres pays d'endémie", "Autres pays d'endemie"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 700, "y_before": 5, "y_after": 55},
    },
    "residence_zone_endemie": {
        "anchor_variants": ["Résidence durant le séjour", "Residence durant le sejour"],
        "options": [
            {"label": "Urbain strict", "variants": ["Urbain strict"]},
            {"label": "Rural", "variants": ["Rural"]},
            {"label": "Itinérant / Mixte", "variants": ["Itinérant / Mixte", "Itinerant / Mixte", "Itinérant/Mixte"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1100, "y_before": 5, "y_after": 65},
    },
    "frequence_sejours": {
        "anchor_variants": ["Fréquence des séjours", "Frequence des sejours"],
        "options": [
            {"label": "1 ou moins", "variants": ["1 ou moins"]},
            {"label": "> 1", "variants": ["> 1"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 800, "y_before": 5, "y_after": 55},
    },
    "patient_adresse": {
        "anchor_variants": ["Patient adressé", "Patient adresse"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 700, "y_before": 5, "y_after": 55},
    },
    "consultation_avant": {
        "anchor_variants": ["consultation médicale avant", "consultation medicale avant", "Y a-t-il eu une consultation"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1000, "y_before": 5, "y_after": 70},
    },
    "etat_clinique": {
        "anchor_variants": ["Etat clinique au moment du diagnostic", "État clinique au moment du diagnostic"],
        "options": [
            {"label": "Accès simple sans vomissements", "variants": ["Accès simple sans vomissements", "Acces simple sans vomissements"]},
            {"label": "Accès simple AVEC vomissements", "variants": ["Accès simple AVEC vomissements", "Acces simple AVEC vomissements"]},
            {"label": "Formes Asymptomatiques et découvertes fortuites", "variants": ["Formes Asymptomatiques et découvertes fortuites"]},
            {"label": "Accès GRAVE", "variants": ["Accès GRAVE", "Acces GRAVE"]},
            {"label": "Paludisme Viscéral évolutif (PVE)", "variants": ["Paludisme Viscéral évolutif", "Paludisme Visceral evolutif", "PVE"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1200, "y_before": 5, "y_after": 140},
    },
    "antecedents_paludisme_3m": {
        "anchor_variants": ["Antécédents de paludisme", "Antecedents de paludisme"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 800, "y_before": 5, "y_after": 45},
    },
    "femme_enceinte": {
        "anchor_variants": ["femme enceinte", "parturiente"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 800, "y_before": 5, "y_after": 45},
    },
    "immunodepression_connue": {
        "anchor_variants": ["immunodépression", "immunodepression"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 900, "y_before": 5, "y_after": 45},
    },
    "paludismes_autochtones": {
        "anchor_variants": ["Paludismes autochtones", 'Paludismes "autochtones"'],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 650, "y_before": 5, "y_after": 45},
    },
    "lame_transmise": {
        "anchor_variants": ["Lame transmise par autre Labo"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 650, "y_before": 5, "y_after": 45},
    },
    "protection_personnelle_oui_non_nsp": {
        "anchor_variants": ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 850, "y_before": 5, "y_after": 45},
    },
    "chimioprophylaxie_oui_non_nsp": {
        "anchor_variants": ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 850, "y_before": 5, "y_after": 45},
    },
    "arret_intolerance_effets_secondaires": {
        "anchor_variants": ["Arrêt de la prise suite à intolérance", "Arret de la prise suite a intolerance"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 900, "y_before": 5, "y_after": 45},
    },
    "utilisation_traitement_curatif_30j": {
        "anchor_variants": ["Utilisation traitement", "Curative du paludisme dans les 30 derniers jours"],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
            {"label": "NSP", "variants": ["NSP"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1100, "y_before": 5, "y_after": 50},
    },
    "prise_en_charge": {
        "anchor_variants": ["Prise en charge & traitement"],
        "options": [
            {"label": "Ambulatoire", "variants": ["Ambulatoire"]},
            {"label": "Hospitalisation", "variants": ["Hospitalisation"]},
            {"label": "Transfert autre hôpital", "variants": ["Transfert autre hôpital", "Transfert autre hopital"]},
            {"label": "Pas de traitement", "variants": ["Pas de traitement"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 1000, "y_before": 5, "y_after": 95},
    },
    "evolution_clinique": {
        "anchor_variants": ["Evolution clinique", "Évolution clinique"],
        "options": [
            {"label": "Guérison", "variants": ["Guérison", "Guerison"]},
            {"label": "DECES", "variants": ["DECES", "Décès", "Deces"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 700, "y_before": 5, "y_after": 45},
    },
    "controle_parasitologique_overall": {
        "anchor_variants": [
            "Contrôle parasitologique P falciparum",
            "Controle parasitologique P falciparum",
            "Contrôle parasitologique",
            "Controle parasitologique",
        ],
        "options": [
            {"label": "Oui", "variants": ["Oui"]},
            {"label": "Non", "variants": ["Non"]},
        ],
        "search_window": {"x_pad_left": 10, "x_pad_right": 800, "y_before": 5, "y_after": 70},
    },
}


def parse_circle_radio_field(words, img_bgr, field_key, debug_dir: Path | None = None, page_stem: str | None = None):
    preset = CIRCLE_FIELD_PRESETS[field_key]

    anchor = None
    for w in words:
        if text_matches(w["text"], preset["anchor_variants"]):
            anchor = w
            break

    result = {
        "field_key": field_key,
        "found_anchor": bool(anchor),
        "anchor_box": None,
        "selected_option": None,
        "options": [],
    }

    if not anchor:
        return result

    anchor_box = [anchor["x1"], anchor["y1"], anchor["x2"], anchor["y2"]]
    result["anchor_box"] = anchor_box

    sw = preset["search_window"]
    local_words = window_words(
        words,
        x1=anchor["x1"] - sw["x_pad_left"],
        x2=anchor["x1"] + sw["x_pad_right"],
        y1=anchor["y1"] - sw["y_before"],
        y2=anchor["y2"] + sw["y_after"],
    )

    debug_items = [
        {"box": anchor_box, "color": (255, 0, 0), "label": "anchor", "thickness": 2}
    ]

    scored = []

    for opt in preset["options"]:
        candidate_words = []
        for w in local_words:
            word_box = [w["x1"], w["y1"], w["x2"], w["y2"]]

            ax1, ay1, ax2, ay2 = anchor_box
            wx1, wy1, wx2, wy2 = word_box
            inter_x1 = max(ax1, wx1)
            inter_y1 = max(ay1, wy1)
            inter_x2 = min(ax2, wx2)
            inter_y2 = min(ay2, wy2)

            overlap = 0.0
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
                w_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))
                overlap = inter / min(a_area, w_area)

            if overlap > 0.5:
                continue

            if text_matches(w["text"], opt["variants"]):
                candidate_words.append(w)

        if not candidate_words:
            result["options"].append({
                "label": opt["label"],
                "found_word": False,
                "state": "missing_word",
            })
            continue

        candidate_words = sorted(candidate_words, key=lambda t: (t["y1"], t["x1"]))
        word = candidate_words[0]

        candidate_boxes = [
            circle_marker_box_for_word(word, left_width=42, right_gap=4, y_pad=8),
            circle_marker_box_for_word(word, left_width=55, right_gap=4, y_pad=10),
            circle_marker_box_for_word(word, left_width=65, right_gap=3, y_pad=12),
            circle_marker_box_for_word(word, left_width=78, right_gap=3, y_pad=12),
            circle_marker_box_for_word(word, left_width=92, right_gap=2, y_pad=14),
            circle_marker_box_for_word(word, left_width=110, right_gap=2, y_pad=16),
        ]

        best_score = None
        best_box = None
        for mb in candidate_boxes:
            crop = crop_box(img_bgr, mb)
            score = score_circle_crop(crop)
            if best_score is None:
                best_score = score
                best_box = mb
                continue

            cur_rank = (
                2 if score["state"] == "selected" else 1 if score["state"] == "unselected" else 0,
                score["dark_ratio"],
                score["center_dark_ratio"],
                score["largest_area"],
            )
            best_rank = (
                2 if best_score["state"] == "selected" else 1 if best_score["state"] == "unselected" else 0,
                best_score["dark_ratio"],
                best_score["center_dark_ratio"],
                best_score["largest_area"],
            )
            if cur_rank > best_rank:
                best_score = score
                best_box = mb

        row = {
            "label": opt["label"],
            "found_word": True,
            "word_text": word["text"],
            "word_box": [word["x1"], word["y1"], word["x2"], word["y2"]],
            "marker_box": best_box,
            "state": best_score["state"],
            "dark_ratio": best_score["dark_ratio"],
            "center_dark_ratio": best_score["center_dark_ratio"],
            "ring_dark_ratio": best_score["ring_dark_ratio"],
            "component_count": best_score["component_count"],
            "largest_area": best_score["largest_area"],
        }

        raw_word = clean_text(word["text"])
        if raw_word.lower().startswith(("o", "0")) and len(raw_word) > 1:
            stripped = raw_word[1:]
            if text_matches(stripped, opt["variants"]):
                row["state"] = "unselected_text"

        result["options"].append(row)
        scored.append(row)

        debug_items.append({
            "box": row["word_box"],
            "color": (0, 255, 0),
            "label": f"text:{opt['label']}",
            "thickness": 2,
        })
        debug_items.append({
            "box": row["marker_box"],
            "color": (0, 165, 255),
            "label": f"marker:{opt['label']}",
            "thickness": 2,
        })

        if debug_dir is not None and page_stem is not None:
            crop = crop_box(img_bgr, row["marker_box"])
            if crop is not None:
                safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", opt["label"])
                crop_path = debug_dir / f"{page_stem}_{field_key}_{safe_label}_marker.png"
                save_image(crop_path, crop)

    selected = [x for x in scored if x["state"] == "selected"]
    strong_unselected = [x for x in scored if x["state"] == "unselected_text"]
    weak_unselected = [x for x in scored if x["state"] == "unselected"]

    if len(selected) == 1:
        result["selected_option"] = selected[0]["label"]
    elif len(selected) > 1:
        selected = sorted(selected, key=lambda x: (-x["center_dark_ratio"], -x["dark_ratio"], -x["largest_area"]))
        result["selected_option"] = selected[0]["label"]
    elif len(preset["options"]) == 2 and len(strong_unselected) == 1:
        unselected_label = strong_unselected[0]["label"]
        other_labels = [o["label"] for o in preset["options"] if o["label"] != unselected_label]
        if len(other_labels) == 1:
            result["selected_option"] = other_labels[0]
            result["selection_inferred_from_other_unselected_text"] = True
    elif len(preset["options"]) == 2 and len(weak_unselected) == 1:
        unselected_label = weak_unselected[0]["label"]
        other_labels = [o["label"] for o in preset["options"] if o["label"] != unselected_label]
        if len(other_labels) == 1:
            result["selected_option"] = other_labels[0]
            result["selection_inferred_from_other_unselected"] = True

    if debug_dir is not None and page_stem is not None:
        dbg = draw_many_boxes(img_bgr, debug_items)
        debug_path = debug_dir / f"{page_stem}_{field_key}_debug.png"
        save_image(debug_path, dbg)

    if result["selected_option"] is None:
        plain_options = [x for x in scored if x["state"] == "unknown"]
        explicit_unselected = [x for x in scored if x["state"] == "unselected_text"]

        if len(explicit_unselected) >= 2 and len(plain_options) == 1:
            result["selected_option"] = plain_options[0]["label"]
            result["selection_inferred_from_plain_vs_unselected_text"] = True

    if result.get("selection_inferred_from_plain_vs_unselected_text"):
        result["confidence"] = "low"
    elif result.get("selected_option") is not None and any(x["state"] == "selected" for x in scored):
        result["confidence"] = "high"
    elif result.get("selected_option") is not None:
        result["confidence"] = "medium"
    else:
        result["confidence"] = "none"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse circle radio fields visually from OCR JSON + page image"
    )
    parser.add_argument("ocr_json_path")
    parser.add_argument("page_image_path")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--field-key",
        required=True,
        choices=sorted(CIRCLE_FIELD_PRESETS.keys()),
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/circle_radio_visual_parser",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    page_image_path = Path(args.page_image_path)
    out_dir = Path(args.output_dir)
    debug_dir = out_dir / "debug"
    ensure_dir(out_dir)
    ensure_dir(debug_dir)

    data = load_json(ocr_json_path)
    data = data.get("res", data)
    words = extract_page_words(data, page_num=args.page_num)

    img_bgr = cv2.imread(str(page_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {page_image_path}")

    page_stem = f"{ocr_json_path.stem}_page{args.page_num}"
    result = parse_circle_radio_field(
        words,
        img_bgr,
        args.field_key,
        debug_dir=debug_dir,
        page_stem=page_stem,
    )

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(page_image_path),
        "page_num": args.page_num,
        "field_key": args.field_key,
        "word_count": len(words),
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_{args.field_key}_circle_radio.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Words loaded: {len(words)}")
    print(f"Saved JSON:   {out_json}")
    print(f"Debug dir:    {debug_dir}")
    print(f"Selected:     {result.get('selected_option')}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
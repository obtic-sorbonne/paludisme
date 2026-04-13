from pathlib import Path
import argparse
import json
import re

from cnr_common import (
    clean_text,
    norm,
    compact_norm,
    parse_prefix_and_text,
    text_matches_option,
    load_ocr_txt,
    extract_page_block,
    postprocess_lines,
    slice_section,
    parse_option_from_lines,
    postprocess_single_choice,
    apply_elimination_heuristic,
    find_first_line_index,
    is_lone_marker,
    line_select_state,
)


def page3_specs():
    return [
        {
            "field": "Espèce(s) Plasmodiale(s)",
            "single_choice": False,
            "start_anchors": ["Espèce(s) Plasmodiale(s)"],
            "end_anchors": ["Hémoglobine", "Hemoglobine"],
            "max_lines": 10,
            "options": [
                ("P falciparum", ["P falciparum"]),
                ("P ovale", ["P ovale", "Povale"]),
                ("Plasmodium spp", ["Plasmodium spp"]),
                ("P vivax", ["P vivax", "Pvivax"]),
                ("P malariae", ["P malariae"]),
            ],
        },
        {
            "field": "Bandelettes",
            "single_choice": True,
            "start_anchors": ["Bandelettes", "Bandelettes (HRP2"],
            "end_anchors": ["Autres techniques", "Commentaire"],
            "max_lines": 10,
            "options": [
                ("Fait", ["Fait"]),
                ("Non fait", ["Non fait"]),
                ("positif Ag P falciparum", ["positif Ag P falciparum"]),
                ("positif Ag commun", ["positif Ag commun"]),
                ("Ag autres espèces", ["Ag autres espèces", "Ag autres especes"]),
            ],
        },
        {
            "field": "Autres techniques",
            "single_choice": False,
            "start_anchors": ["Autres techniques"],
            "end_anchors": ["Commentaire", "Lame transmise"],
            "max_lines": 8,
            "options": [
                ("PCR", ["PCR"]),
                ("QBC", ["QBC"]),
                ("Sérologie", ["Sérologie", "Serologie"]),
                ("Autres", ["Autres"]),
            ],
        },
        {
            "field": "Lame transmise par autre Labo",
            "single_choice": True,
            "start_anchors": ["Lame transmise par autre Labo"],
            "end_anchors": ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"],
            "max_lines": 6,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
            ],
        },
        {
            "field": "Protection Personnelle Anti-Moustiques",
            "single_choice": True,
            "start_anchors": ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"],
            "end_anchors": ["Répulsifs cutanés", "Repulsifs cutanes"],
            "max_lines": 8,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
        {
            "field": "Répulsifs cutanés",
            "single_choice": True,
            "start_anchors": ["Répulsifs cutanés", "Repulsifs cutanes"],
            "end_anchors": ["Moustiquaires de lit"],
            "max_lines": 6,
            "options": [
                ("Sans autre indication", ["Sans autre indication"]),
                ("Régulier", ["Régulier", "Regulier"]),
                ("Episodique", ["Episodique"]),
            ],
        },
        {
            "field": "Moustiquaires de lit",
            "single_choice": True,
            "start_anchors": ["Moustiquaires de lit"],
            "end_anchors": ["Autres, préciser", "Autres, preciser", "Chimioprophylaxie"],
            "max_lines": 6,
            "options": [
                ("Sans autre indication", ["Sans autre indication"]),
                ("Régulier", ["Régulier", "Regulier"]),
                ("Episodique", ["Episodique"]),
            ],
        },
        {
            "field": "Chimioprophylaxie utilisée",
            "single_choice": True,
            "start_anchors": ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"],
            "end_anchors": ["Date de la dernière prise", "Date de la demière prise", "Arrêt"],
            "max_lines": 35,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
                ("Atovaquone + Proguanil (Malarone®)", ["Atovaquone + Proguanil", "Malarone"]),
                ("Méfloquine (Lariam®)", ["Méfloquine", "Lariam", "Mefloquine"]),
                ("Doxycycline (Doxypalu®)", ["Doxycycline", "Doxypalu"]),
                ("Chloroquine + Proguanil (ou Savarine®)", ["Chloroquine + Proguanil", "Savarine"]),
                ("Chloroquine (Nivaquine®)", ["Chloroquine (Nivaquine®)", "Nivaquine"]),
                ("Proguanil (Paludrine®)", ["Proguanil (Paludrine®)", "Paludrine"]),
                ("Autre, préciser", ["Autre, préciser", "Autre, preciser"]),
            ],
        },
        {
            "field": "Arrêt de la prise suite à intolérance/effet(s) secondaire(s)",
            "single_choice": True,
            "start_anchors": ["Arrêt de la prise suite à intolérance", "Arret de la prise suite a intolerance"],
            "end_anchors": None,
            "max_lines": 6,
            "options": [
                ("Oui", ["Oui"]),
                ("Non", ["Non"]),
                ("NSP", ["NSP"]),
            ],
        },
    ]


def parse_protection_personnelle_block(lines):
    result = {
        "field": "Protection Personnelle Anti-Moustiques",
        "found": False,
        "selected": None,
        "details": []
    }

    start_idx = find_first_line_index(lines, ["Protection Personnelle Anti-Moustiques", "Protection Personnelle Anti-"])
    if start_idx is None:
        return result

    result["found"] = True

    block_top = lines[start_idx:start_idx + 8]
    for line in block_top:
        prefix, content = parse_prefix_and_text(line)
        txt = clean_text(content if prefix else line)
        if norm(txt) in {"oui", "non", "nsp"} and prefix == "X":
            result["selected"] = txt

    row_specs = [
        ("Répulsifs cutanés", ["Répulsifs cutanés", "Repulsifs cutanes"]),
        ("Moustiquaires de lit", ["Moustiquaires de lit"]),
        ("Autres, préciser", ["Autres, préciser", "Autres, preciser"]),
    ]

    row_options = ["Sans autre indication", "Régulier", "Episodique"]

    for row_name, anchors in row_specs:
        idx = find_first_line_index(lines, anchors)
        if idx is None:
            continue

        row_slice = lines[idx: idx + 6]
        selected = None
        found_options = []

        for line in row_slice:
            prefix, content = parse_prefix_and_text(line)
            txt = clean_text(content if prefix else line)

            for opt in row_options:
                if text_matches_option(txt, opt):
                    found_options.append(opt)
                    if prefix == "X":
                        selected = opt

        result["details"].append({
            "item": row_name,
            "selection": selected,
            "options_found": found_options
        })

    return result


def parse_chimioprophylaxie_block(lines):
    result = {
        "field": "Chimioprophylaxie utilisée",
        "found": False,
        "selected": None,
        "table_results": []
    }

    start_idx = find_first_line_index(lines, ["Chimioprophylaxie utilisée", "Chimioprophylaxie utilisee"])
    if start_idx is None:
        return result

    result["found"] = True

    heading_slice = lines[start_idx:start_idx + 8]
    for line in heading_slice:
        prefix, content = parse_prefix_and_text(line)
        txt = clean_text(content if prefix else line)
        if norm(txt) in {"oui", "non", "nsp"} and prefix == "X":
            result["selected"] = txt

    medicines = [
        "Atovaquone + Proguanil (Malarone®)",
        "Méfloquine (Lariam®)",
        "Doxycycline (Doxypalu®)",
        "Chloroquine + Proguanil (ou Savarine®)",
        "Chloroquine (Nivaquine®)",
        "Proguanil (Paludrine®)",
        "Autre, préciser",
    ]

    medicine_variants = {
        "Atovaquone + Proguanil (Malarone®)": ["Atovaquone + Proguanil (Malarone®)", "Atovaquone + Proguanil", "Malarone"],
        "Méfloquine (Lariam®)": ["Méfloquine (Lariam®)", "Méfloquine", "Lariam", "Mefloquine"],
        "Doxycycline (Doxypalu®)": ["Doxycycline (Doxypalu®)", "Doxycycline", "Doxypalu"],
        "Chloroquine + Proguanil (ou Savarine®)": ["Chloroquine + Proguanil (ou Savarine®)", "Chloroquine + Proguanil", "Savarine"],
        "Chloroquine (Nivaquine®)": ["Chloroquine (Nivaquine®)", "Chloroquine", "Nivaquine"],
        "Proguanil (Paludrine®)": ["Proguanil (Paludrine®)", "Proguanil", "Paludrine"],
        "Autre, préciser": ["Autre, préciser", "Autre, preciser"],
    }

    end_idx = find_first_line_index(
        lines[start_idx:],
        ["Date de la dernière prise", "Date de la derniere prise", "Arrêt de la prise suite à intolérance"]
    )
    if end_idx is None:
        block = lines[start_idx:start_idx + 40]
    else:
        block = lines[start_idx:start_idx + end_idx]

    med_positions = []
    for i, line in enumerate(block):
        for med in medicines:
            if any(text_matches_option(line, var) for var in medicine_variants[med]):
                med_positions.append((i, med))
                break

    dedup = []
    seen_meds = set()
    for idx, med in med_positions:
        if med not in seen_meds:
            dedup.append((idx, med))
            seen_meds.add(med)
    med_positions = dedup

    def looks_like_free_value(txt: str) -> bool:
        t = clean_text(txt)
        nt = norm(t)
        if not t:
            return False
        if nt in {"x", "o", "oui", "non", "nsp"}:
            return False
        if any(any(text_matches_option(t, var) for var in medicine_variants[m]) for m in medicines):
            return False
        if "date de la derniere prise" in nt or "date de la dernière prise" in nt:
            return False
        if "si pas de date" in nt or "arrêt de la prise suite" in nt or "arret de la prise suite" in nt:
            return False
        return True

    for pos_idx, (idx, med) in enumerate(med_positions):
        next_idx = med_positions[pos_idx + 1][0] if pos_idx + 1 < len(med_positions) else len(block)
        row_slice = block[idx:next_idx]

        selected_label = None

        for line in row_slice[:3]:
            prefix, content = parse_prefix_and_text(line)
            if prefix == "X":
                if text_matches_option(content, med) or text_matches_option(line, med):
                    selected_label = "Régulier"
                    break

        if selected_label is None:
            for line in row_slice[:3]:
                txt = clean_text(line)
                if looks_like_free_value(txt):
                    selected_label = txt
                    break

        if selected_label is None:
            for j in range(1, min(len(row_slice), 4)):
                txt = clean_text(row_slice[j])
                if looks_like_free_value(txt):
                    prev = clean_text(row_slice[j - 1])
                    mt = is_lone_marker(prev)
                    if mt == "X":
                        selected_label = txt
                        break

        if selected_label:
            result["table_results"].append({
                "medicine": med,
                "selection": selected_label
            })

    return result


def parse_bandelettes_block(lines):
    result = {
        "field": "Bandelettes",
        "found": False,
        "status": None,
        "selected_results": [],
        "all_results_found": [],
    }

    start_idx = find_first_line_index(lines, ["Bandelettes", "Bandelettes (HRP2"])
    if start_idx is None:
        return result

    result["found"] = True

    end_idx = None
    for j in range(start_idx + 1, min(len(lines), start_idx + 15)):
        ln = norm(lines[j])
        if "autres techniques" in ln or "commentaire" in ln:
            end_idx = j
            break

    if end_idx is None:
        end_idx = min(len(lines), start_idx + 15)

    block = lines[start_idx:end_idx]

    for line in block:
        ln = norm(line)
        if "non fait" in ln:
            result["status"] = "Non fait"
            break

    if result["status"] is None:
        for line in block:
            ln = norm(line)
            if "fait" in ln:
                result["status"] = "Fait"
                break

    option_specs = [
        ("positif Ag P falciparum", ["positif Ag P falciparum"]),
        ("positif Ag commun", ["positif Ag commun"]),
        ("Ag autres espèces", ["Ag autres espèces", "Ag autres especes"]),
    ]

    found_exact = []
    for canonical, variants in option_specs:
        for line in block:
            ln = clean_text(line)
            if any(text_matches_option(ln, v) for v in variants):
                found_exact.append(canonical)
                break

    dedup = []
    for x in found_exact:
        if x not in dedup:
            dedup.append(x)

    result["all_results_found"] = dedup

    if result["status"] == "Fait" and dedup:
        result["selected_results"] = [dedup[0]]

    return result


def parse_examens_parasitologiques_block(lines):
    """
    Structured parser for the fixed 'Examens parasitologiques' block.
    """

    def _n(s):
        return norm(clean_text(s))

    def _find_block_start():
        anchors = [
            "frottis mince",
            "goutte épaisse",
            "goutte epaisse",
            "bandelettes",
            "autres techniques",
        ]
        for i, line in enumerate(lines):
            ln = _n(line)
            if any(a in ln for a in anchors):
                return i
        return None

    def _find_block_end(start_idx):
        end_anchors = [
            "lame transmise par autre labo",
            "protection personnelle anti-moustiques",
            "protection personnelle anti-",
            "chimioprophylaxie utilisée",
            "chimioprophylaxie utilisee",
        ]
        for j in range(start_idx + 1, min(len(lines), start_idx + 80)):
            ln = _n(lines[j])
            if any(a in ln for a in end_anchors):
                return j
        return min(len(lines), start_idx + 80)

    def _collect_section(block, section_anchors):
        idx = None
        for i, line in enumerate(block):
            ln = _n(line)
            if any(a in ln for a in section_anchors):
                idx = i
                break
        if idx is None:
            return []

        next_anchor_idx = None
        next_groups = [
            ["frottis mince"],
            ["goutte épaisse", "goutte epaisse"],
            ["bandelettes"],
            ["autres techniques"],
            ["lame transmise par autre labo"],
            ["protection personnelle anti-moustiques", "protection personnelle anti-"],
        ]

        for j in range(idx + 1, len(block)):
            ln = _n(block[j])
            if any(any(a in ln for a in group) for group in next_groups):
                next_anchor_idx = j
                break

        if next_anchor_idx is None:
            next_anchor_idx = len(block)

        return block[idx:next_anchor_idx]

    def _repair_frottis_goutte_boundary(frottis_lines, goutte_lines):
        """
        OCR sometimes leaves 'Non fait' and 'pour' at the end of frottis_lines
        even though they belong to the goutte épaisse subsection.
        """
        if not frottis_lines:
            return frottis_lines, goutte_lines

        repaired_frottis = list(frottis_lines)
        repaired_goutte = list(goutte_lines)

        moved_tail = []

        while repaired_frottis:
            tail = clean_text(repaired_frottis[-1])
            ntail = _n(tail)

            if ntail in {"non fait", "pour"}:
                moved_tail.append(repaired_frottis.pop())
                continue
            break

        moved_tail.reverse()

        if moved_tail:
            repaired_goutte = moved_tail + repaired_goutte

        return repaired_frottis, repaired_goutte

    def _extract_status(section_lines, keep_presence_text=False):
        if not section_lines:
            return None

        joined_norm = " | ".join(_n(x) for x in section_lines if clean_text(x))

        if "non fait" in joined_norm and not keep_presence_text:
            return "Non fait"

        if keep_presence_text:
            for line in section_lines:
                txt = clean_text(line)
                ln = _n(txt)
                if (
                    "presence de trophozoites" in ln
                    or "présence de trophozoites" in ln
                    or "schizontes" in ln
                ):
                    return txt

        for line in section_lines:
            prefix, content = parse_prefix_and_text(line)
            txt = clean_text(content if prefix else line)
            ln = _n(txt)

            if prefix == "X" and ln == "non":
                return "Non fait"
            if prefix == "X" and ln == "oui":
                return "Fait"

        if re.search(r"(^|\s)fait($|\s)", joined_norm):
            return "Fait"

        return None

    def _extract_percent_values(section_lines):
        vals = []

        for i, line in enumerate(section_lines):
            txt = clean_text(line)

            for m in re.finditer(r"\b\d{1,2}(?:[.,]\d+)?\s*%", txt):
                raw = clean_text(m.group(0)).replace(" ", "")
                val = raw[:-1].replace(".", ",")
                if val not in vals:
                    vals.append(val)

            nums = re.findall(r"\b\d{1,2}(?:[.,]\d+)?\b", txt)
            if nums:
                local_window = " ".join(
                    _n(section_lines[j])
                    for j in range(max(0, i - 1), min(len(section_lines), i + 2))
                )
                if "%" in local_window or "pourcent" in local_window:
                    for num in nums:
                        val = num.replace(".", ",")
                        if val not in vals:
                            vals.append(val)

            if "%" in txt:
                for j in [i - 1, i + 1]:
                    if 0 <= j < len(section_lines):
                        nearby = clean_text(section_lines[j])
                        nearby_nums = re.findall(r"\b\d{1,2}(?:[.,]\d+)?\b", nearby)
                        for num in nearby_nums:
                            val = num.replace(".", ",")
                            if val not in vals:
                                vals.append(val)

        return vals

    def _extract_count_values(section_lines, status=None):
        if not section_lines:
            return []

        if status == "Non fait":
            return []

        vals = []
        reject_fragments = [
            "1000 gb",
            "pour",
            "si fait",
            "densité parasitaire",
            "densite parasitaire",
            "statut",
            "%",
        ]

        for line in section_lines:
            txt = clean_text(line)
            ln = _n(txt)

            if any(r in ln for r in reject_fragments):
                continue

            for m in re.finditer(r"\b\d{4,7}\b", txt):
                v = m.group(0)
                if v not in vals:
                    vals.append(v)

        return vals

    def _extract_per_1000_gb_values(section_lines, status=None):
        if not section_lines:
            return []

        if status == "Non fait":
            return []

        vals = []

        for line in section_lines:
            txt = clean_text(line)
            ln = _n(txt)

            m = re.search(r"\b(\d{1,4})\b.*\b1000\s*gb\b", ln)
            if m:
                v = m.group(1)
                if v not in vals:
                    vals.append(v)

        return vals

    def _extract_selected_options(section_lines, option_specs):
        selected = []
        found = []

        for canonical, variants in option_specs:
            hit_found = False
            hit_selected = False

            for line in section_lines:
                prefix, content = parse_prefix_and_text(line)
                check_txt = content if prefix else line

                if any(text_matches_option(check_txt, v) for v in variants):
                    hit_found = True
                    if prefix == "X":
                        hit_selected = True

            if hit_found and canonical not in found:
                found.append(canonical)
            if hit_selected and canonical not in selected:
                selected.append(canonical)

        return selected, found

    start_idx = _find_block_start()
    if start_idx is None:
        return {
            "field": "Examens parasitologiques",
            "found": False,
            "frottis_mince": {},
            "goutte_epaisse": {},
            "bandelettes": {},
            "autres_techniques": {},
        }

    end_idx = _find_block_end(start_idx)
    block = lines[start_idx:end_idx]

    frottis_lines = _collect_section(block, ["frottis mince"])
    goutte_lines = _collect_section(block, ["goutte épaisse", "goutte epaisse"])
    frottis_lines, goutte_lines = _repair_frottis_goutte_boundary(frottis_lines, goutte_lines)

    bandelette_lines = _collect_section(block, ["bandelettes"])
    autres_lines = _collect_section(block, ["autres techniques"])

    bandelette_option_specs = [
        ("positif Ag P falciparum", ["positif Ag P falciparum"]),
        ("positif Ag commun", ["positif Ag commun"]),
        ("Ag autres espèces", ["Ag autres espèces", "Ag autres especes"]),
    ]

    autres_option_specs = [
        ("PCR", ["PCR"]),
        ("QBC", ["QBC"]),
        ("Sérologie", ["Sérologie", "Serologie"]),
        ("Autres", ["Autres"]),
    ]

    bandelette_selected, bandelette_found = _extract_selected_options(
        bandelette_lines, bandelette_option_specs
    )
    autres_selected, autres_found = _extract_selected_options(
        autres_lines, autres_option_specs
    )

    frottis_status = _extract_status(frottis_lines[:6], keep_presence_text=True)
    goutte_status = _extract_status(goutte_lines[:8], keep_presence_text=False)

    if goutte_status is None:
        goutte_joined = " | ".join(_n(x) for x in goutte_lines if clean_text(x))
        if "non fait" in goutte_joined:
            goutte_status = "Non fait"

    bandelette_status = _extract_status(bandelette_lines, keep_presence_text=False)

    return {
        "field": "Examens parasitologiques",
        "found": True,
        "frottis_mince": {
            "status": frottis_status,
            "percent_values": _extract_percent_values(frottis_lines),
            "count_values": _extract_count_values(frottis_lines, frottis_status),
            "raw_lines": frottis_lines,
        },
        "goutte_epaisse": {
            "status": goutte_status,
            "per_1000_gb_values": _extract_per_1000_gb_values(goutte_lines, goutte_status),
            "count_values": _extract_count_values(goutte_lines, goutte_status),
            "raw_lines": goutte_lines,
        },
        "bandelettes": {
            "status": bandelette_status,
            "selected_options": bandelette_selected,
            "found_options": bandelette_found,
            "raw_lines": bandelette_lines,
        },
        "autres_techniques": {
            "selected_options": autres_selected,
            "found_options": autres_found,
            "raw_lines": autres_lines,
        },
    }


def parse_page3_lab_values_block(lines):
    def _clean(s):
        return clean_text(s)

    def _n(s):
        return norm(_clean(s))

    def _find_first_anchor_idx(anchor_variants):
        for i, line in enumerate(lines):
            ln = _n(line)
            for a in anchor_variants:
                if _n(a) in ln:
                    return i
        return None

    def _extract_numeric_from_text(txt):
        txt = _clean(txt).replace(",", ".")
        return re.findall(r"\b\d+(?:\.\d+)?\b", txt)

    def _extract_value_near_anchor(anchor_variants, window_before=0, window_after=4):
        idx = _find_first_anchor_idx(anchor_variants)
        if idx is None:
            return None

        start = max(0, idx - window_before)
        end = min(len(lines), idx + window_after + 1)

        candidates = []

        for j in range(start, end):
            txt = _clean(lines[j])
            if not txt:
                continue

            if j == idx and ":" in txt:
                right = _clean(txt.split(":", 1)[1])
                nums = _extract_numeric_from_text(right)
                if nums:
                    candidates.extend(nums)

            nums = _extract_numeric_from_text(txt)
            if nums:
                candidates.extend(nums)

        if not candidates:
            return None

        for v in candidates:
            try:
                float(v)
            except Exception:
                continue
            return v.replace(".", ",")

        return None

    result = {
        "field": "Valeurs biologiques",
        "found": False,
        "Hemoglobine (g/l)": None,
        "GR (tera/l)": None,
        "GB (giga/l)": None,
        "Plaquettes (giga/l)": None,
    }

    bio_anchor_idx = _find_first_anchor_idx([
        "Hémoglobine",
        "Hemoglobine",
        "GR",
        "GB",
        "Plaquettes",
    ])
    if bio_anchor_idx is None:
        return result

    hgb = _extract_value_near_anchor(["Hémoglobine", "Hemoglobine"])
    gr = _extract_value_near_anchor(["GR"])
    gb = _extract_value_near_anchor(["GB"])
    plaquettes = _extract_value_near_anchor(["Plaquettes"])

    def valid_hgb(v):
        if v is None:
            return False
        try:
            x = float(v.replace(",", "."))
            return 20 <= x <= 250
        except Exception:
            return False

    def valid_gr(v):
        if v is None:
            return False
        try:
            x = float(v.replace(",", "."))
            return 1 <= x <= 8
        except Exception:
            return False

    def valid_gb(v):
        if v is None:
            return False
        try:
            x = float(v.replace(",", "."))
            return 0.1 <= x <= 100
        except Exception:
            return False

    def valid_plaq(v):
        if v is None:
            return False
        try:
            x = float(v.replace(",", "."))
            return 1 <= x <= 1500
        except Exception:
            return False

    if valid_hgb(hgb):
        result["Hemoglobine (g/l)"] = hgb
    if valid_gr(gr):
        result["GR (tera/l)"] = gr
    if valid_gb(gb):
        result["GB (giga/l)"] = gb
    if valid_plaq(plaquettes):
        result["Plaquettes (giga/l)"] = plaquettes

    found_count = sum(
        1 for k in [
            "Hemoglobine (g/l)",
            "GR (tera/l)",
            "GB (giga/l)",
            "Plaquettes (giga/l)",
        ]
        if result[k] is not None
    )

    if found_count >= 3:
        result["found"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse CNR page 3 options from OCR txt")
    parser.add_argument("ocr_txt_path")
    parser.add_argument("--page-num", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page3_ocr_text_parser",
    )
    args = parser.parse_args()

    ocr_txt_path = Path(args.ocr_txt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_file_lines = load_ocr_txt(ocr_txt_path)
    raw_page_lines = extract_page_block(raw_file_lines, page_num=args.page_num)

    replacements = [
        ("ONSP", "O NSP"),
        ("Povale", "P ovale"),
        ("Pvivax", "P vivax"),
        ("OOui", "O Oui"),
        ("O ui", "O Oui"),
    ]
    lines = postprocess_lines(raw_page_lines, replacements=replacements)

    sections = []
    field_results = []
    custom_blocks = []

    for spec in page3_specs():
        sec = slice_section(
            lines=lines,
            start_anchors=spec["start_anchors"],
            end_anchors=spec["end_anchors"],
            max_lines=spec["max_lines"],
        )

        if sec is None:
            sections.append({"field": spec["field"], "found": False})
            field_results.append({"field": spec["field"], "found": False, "options": []})
            continue

        sections.append({
            "field": spec["field"],
            "found": True,
            "anchor_line": sec["anchor_line"],
            "start_idx": sec["start_idx"],
            "end_idx": sec["end_idx"],
            "lines": sec["lines"],
        })

        option_results = []
        for canonical, variants in spec["options"]:
            option_results.append(parse_option_from_lines(sec["lines"], canonical, variants))

        option_results = postprocess_single_choice(option_results)
        option_results = apply_elimination_heuristic(option_results, spec.get("single_choice", False))
        selected_options = [x["option"] for x in option_results if x["selected"]]

        field_results.append({
            "field": spec["field"],
            "found": True,
            "selected_options": selected_options,
            "options": option_results,
        })

    custom_blocks.append(parse_examens_parasitologiques_block(lines))
    custom_blocks.append(parse_protection_personnelle_block(lines))
    custom_blocks.append(parse_chimioprophylaxie_block(lines))
    custom_blocks.append(parse_bandelettes_block(lines))
    custom_blocks.append(parse_page3_lab_values_block(lines))

    result = {
        "ocr_txt_path": str(ocr_txt_path),
        "page_num": args.page_num,
        "lines_count": len(lines),
        "ocr_lines_preview": lines[:150],
        "sections": sections,
        "field_results": field_results,
        "custom_blocks": custom_blocks,
    }

    out_json = out_dir / f"{ocr_txt_path.stem}_page{args.page_num}_ocr_parsed.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lines loaded:       {len(lines)}")
    print(f"Fields parsed:      {len(field_results)}")
    print(f"Saved JSON:         {out_json}")


if __name__ == "__main__":
    main()
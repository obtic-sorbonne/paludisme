from pathlib import Path
import argparse
import json
import re


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root(d: dict):
    return d.get("res", d)


def clean_text(text):
    text = str(text).strip()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def norm(s: str):
    return " ".join(
        s.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ï", "i")
        .replace("î", "i")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("’", "'")
        .split()
    )


def get_paddle_lines(paddle_data: dict):
    root = get_root(paddle_data)
    ocr = root.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts", [])
    boxes = ocr.get("rec_boxes", [])

    lines = []
    for t, b in zip(texts, boxes):
        if not isinstance(t, str):
            continue
        t = clean_text(t)
        if not t:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        lines.append({
            "text": t,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })
    return lines


def group_rows(lines, y_thresh=18):
    if not lines:
        return []

    lines = sorted(lines, key=lambda x: (x["cy"], x["x1"]))
    rows = []
    current = [lines[0]]
    current_y = lines[0]["cy"]

    for line in lines[1:]:
        if abs(line["cy"] - current_y) <= y_thresh:
            current.append(line)
            current_y = sum(l["cy"] for l in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda x: x["x1"]))
            current = [line]
            current_y = line["cy"]

    rows.append(sorted(current, key=lambda x: x["x1"]))
    return rows


def merge_close_in_row(row, x_gap=18):
    if not row:
        return []

    merged = []
    cur = row[0].copy()

    for item in row[1:]:
        if item["x1"] - cur["x2"] <= x_gap:
            cur["text"] = f"{cur['text']} {item['text']}"
            cur["x2"] = max(cur["x2"], item["x2"])
            cur["y1"] = min(cur["y1"], item["y1"])
            cur["y2"] = max(cur["y2"], item["y2"])
            cur["cx"] = (cur["x1"] + cur["x2"]) / 2
            cur["cy"] = (cur["y1"] + cur["y2"]) / 2
        else:
            merged.append(cur)
            cur = item.copy()

    merged.append(cur)
    return merged


def row_text(row):
    return clean_text(" | ".join(x["text"] for x in row if clean_text(x["text"])))


def looks_like_metadata(text):
    t = norm(text)
    keys = [
        "nom patient", "date / heure", "date naissance", "prescripteur",
        "adresse", "patient adresse", "copie a", "echantillon", "prelevement",
        "demande", "resultats d'une demande", "consultit", "page 1 sur 2",
        "page 2 sur 2", "page 1 sur 1", "hopital robert debre", "valide par",
        "requestresults.aspx", "tel:"
    ]
    return any(k in t for k in keys)


def looks_like_header(text):
    t = norm(text)
    keys = ["description", "resultat", "unite", "valeurs normales", "val"]
    return sum(1 for k in keys if k in t) >= 2


def count_nonempty(row):
    return sum(1 for x in row if clean_text(x))


def structured_rows_quality(rows):
    if not rows:
        return 0.0

    good = 0
    for row in rows:
        desc = clean_text(row[0]) if len(row) > 0 else ""
        payload = any(clean_text(x) for x in row[1:])
        if desc and payload:
            good += 1
        elif count_nonempty(row) >= 3:
            good += 1

    return round(good / max(len(rows), 1), 3)


def hematology_score(text):
    t = norm(text)
    keys = [
        "erythrocytes", "hemoglobine", "hematocrite", "leucocytes",
        "plaquettes", "lymphocytes", "monocytes", "poly neutrophiles",
        "poly eosinophiles", "poly basophiles", "myelocytes",
        "metamyelocytes", "blastes", "plasmocytes"
    ]
    return sum(1 for k in keys if k in t)


def biochemistry_score(text):
    t = norm(text)
    keys = [
        "hemolyse", "ictere", "lipemie", "sodium", "potassium", "chlore",
        "bicarbonates", "proteines", "uree", "creatinine", "glycemie",
        "bilirubine", "crp", "ldh", "procalcitonine", "phosphatases alcalines",
        "asat", "alat", "ggt", "haptoglobine"
    ]
    return sum(1 for k in keys if k in t)


def detect_family(lines):
    joined = "\n".join(clean_text(x["text"]) for x in lines if clean_text(x["text"]))
    h = hematology_score(joined)
    b = biochemistry_score(joined)

    if h == 0 and b == 0:
        return "unknown"
    return "hematology" if h >= b else "biochemistry"


def simple_assign_5cols(row):
    items = [clean_text(x["text"]) for x in row if clean_text(x["text"])]
    items = items[:5]
    while len(items) < 5:
        items.append("")
    return items


def rescue_hematology(lines):
    rows = group_rows(lines, y_thresh=18)
    rows = [merge_close_in_row(r, x_gap=20) for r in rows]

    out = []
    for row in rows:
        txt = row_text(row)
        if not txt or looks_like_metadata(txt) or looks_like_header(txt):
            continue

        # description + value + optional unit
        items = [clean_text(x["text"]) for x in row if clean_text(x["text"])]
        if len(items) < 2:
            continue

        desc = items[0]
        result = items[1] if len(items) > 1 else ""
        unit = items[2] if len(items) > 2 else ""
        normal = items[3] if len(items) > 3 else ""
        val = items[4] if len(items) > 4 else ""

        dn = norm(desc)
        if any(k in dn for k in [
            "erythrocytes", "hemoglobine", "hematocrite", "leucocytes", "plaquettes",
            "lymphocytes", "monocytes", "poly", "myelocytes", "metamyelocytes",
            "blastes", "plasmocytes", "anomalies morph"
        ]):
            out.append([desc, result, unit, normal, val])

    return out


def rescue_biochemistry(lines):
    rows = group_rows(lines, y_thresh=18)
    rows = [merge_close_in_row(r, x_gap=20) for r in rows]

    out = []
    for row in rows:
        txt = row_text(row)
        if not txt or looks_like_metadata(txt) or looks_like_header(txt):
            continue

        items = [clean_text(x["text"]) for x in row if clean_text(x["text"])]
        if len(items) < 2:
            continue

        desc = items[0]
        dn = norm(desc)

        if not any(k in dn for k in [
            "hemolyse", "ictere", "lipemie", "sodium", "potassium", "chlore",
            "bicarbonates", "proteines", "uree", "creatinine", "glycemie",
            "bilirubine", "crp", "ldh", "procalcitonine", "phosphatases alcalines",
            "asat", "alat", "ggt", "haptoglobine"
        ]):
            continue

        result = items[1] if len(items) > 1 else ""
        unit = items[2] if len(items) > 2 else ""
        ref_min = items[3] if len(items) > 3 else ""
        ref_max = items[4] if len(items) > 4 else ""

        out.append([desc, result, unit, ref_min, ref_max])

    return out


def save_rescue_output(out_path: Path, family: str, rows, failure_reason: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Rescue family: {family}\n")
        f.write(f"Failure reason: {failure_reason}\n\n")
        f.write("=== CORE TABLE (RESCUE) ===\n\n")
        for row in rows:
            f.write(" | ".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rescue lab extraction from OCR when hybrid extraction fails")
    parser.add_argument("paddle_json")
    parser.add_argument("--failure-reason", default="Hybrid file was not created")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables",
    )
    args = parser.parse_args()

    paddle_json = Path(args.paddle_json)
    out_dir = Path(args.output_dir)

    data = load_json(paddle_json)
    lines = get_paddle_lines(data)

    family = detect_family(lines)

    hem_rows = rescue_hematology(lines)
    bio_rows = rescue_biochemistry(lines)

    hem_q = structured_rows_quality(hem_rows)
    bio_q = structured_rows_quality(bio_rows)

    chosen_family = "unknown"
    chosen_rows = []

    if hem_q >= bio_q and hem_q >= 0.45 and len(hem_rows) >= 3:
        chosen_family = "hematology"
        chosen_rows = hem_rows
    elif bio_q > hem_q and bio_q >= 0.45 and len(bio_rows) >= 2:
        chosen_family = "biochemistry"
        chosen_rows = bio_rows

    if not chosen_rows:
        print("No usable rescue table could be built.")
        print(f"Hematology quality: {hem_q}")
        print(f"Biochemistry quality: {bio_q}")
        return 1

    out_file = out_dir / f"{paddle_json.stem}_hybrid.txt"
    save_rescue_output(out_file, chosen_family, chosen_rows, args.failure_reason)

    print(f"Rescue family used: {chosen_family}")
    print(f"Hematology quality: {hem_q}")
    print(f"Biochemistry quality: {bio_q}")
    print(f"Saved rescue hybrid output: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
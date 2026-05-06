from pathlib import Path
import argparse
import json

import cv2

from parse_page4_controle_parasito_visual import (
    load_json as load_visual_json,
    extract_page_words as extract_visual_words,
    parse_controle_block as parse_controle_block_circle,
)
from parse_page4_controle_parasito_square import (
    load_json as load_square_json,
    extract_page_words as extract_square_words,
    parse_controle_block as parse_controle_block_square,
)


def detect_template_type(words):
    texts = [w["text"] for w in words]

    has_fait_header = any("Fait" in t for t in texts)
    has_temp_header = any("Température" in t or "Temperature" in t for t in texts)
    has_dense_square_table = any("≤100" in t or "101-10 000" in t or "> 10 000" in t or "101-10000" in t for t in texts)

    if has_fait_header and has_temp_header and has_dense_square_table:
        return "square"

    return "circle"


def main():
    parser = argparse.ArgumentParser(description="Route contrôle parasitologique page to square or circle parser")
    parser.add_argument("ocr_json_path")
    parser.add_argument("--page-image-path", required=True)
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/page4_controle_parasito_router",
    )
    args = parser.parse_args()

    ocr_json_path = Path(args.ocr_json_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_visual_json(ocr_json_path)
    data = data.get("res", data)
    words = extract_visual_words(data, page_num=args.page_num)

    img_bgr = cv2.imread(str(args.page_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.page_image_path}")

    template_type = detect_template_type(words)

    if template_type == "square":
        result = parse_controle_block_square(words, img_bgr=img_bgr)
    else:
        result = parse_controle_block_circle(words, img_bgr=img_bgr)

    final = {
        "ocr_json_path": str(ocr_json_path),
        "page_image_path": str(args.page_image_path),
        "page_num": args.page_num,
        "template_type": template_type,
        "result": result,
    }

    out_json = out_dir / f"{ocr_json_path.stem}_page{args.page_num}_controle_parasito_router.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Template detected:  {template_type}")
    print(f"Saved JSON:         {out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
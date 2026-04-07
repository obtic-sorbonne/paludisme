import json
from pathlib import Path
import argparse


def walk(obj, texts, html_blocks):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"text", "label", "content"} and isinstance(v, str) and v.strip():
                texts.append(v.strip())

            if k in {"html", "pred_html"} and isinstance(v, str) and v.strip():
                html_blocks.append(v)

            if k in {"rec_texts", "texts"} and isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        texts.append(item.strip())

            walk(v, texts, html_blocks)

    elif isinstance(obj, list):
        for item in obj:
            walk(item, texts, html_blocks)


def main():
    parser = argparse.ArgumentParser(description="Parse Paddle table JSON into readable text")
    parser.add_argument("json_path", help="Full path to Paddle JSON file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table_txt",
        help="Directory to save readable output",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    html_blocks = []
    walk(data, texts, html_blocks)

    seen = set()
    cleaned = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    out_file = out_dir / f"{json_path.stem}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== READABLE TEXT EXTRACT ===\n\n")
        for line in cleaned:
            f.write(line + "\n")

        if html_blocks:
            f.write("\n\n=== HTML TABLE BLOCKS ===\n\n")
            for i, block in enumerate(html_blocks, start=1):
                f.write(f"[HTML BLOCK {i}]\n")
                f.write(block + "\n\n")

    print(f"Saved: {out_file}")
    print(f"Extracted {len(cleaned)} text items")
    print(f"Found {len(html_blocks)} HTML block(s)")


if __name__ == "__main__":
    main()
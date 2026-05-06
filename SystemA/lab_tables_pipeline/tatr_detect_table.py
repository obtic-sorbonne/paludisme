from pathlib import Path
import argparse
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch


def pdf_page_to_pil(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.n == 4 else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    doc.close()
    return img


def main():
    parser = argparse.ArgumentParser(description="Detect tables in a PDF page with TATR")
    parser.add_argument("pdf_path", help="Full path to PDF")
    parser.add_argument("--page-index", type=int, default=0, help="Page index (0-based)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/tatr",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = pdf_page_to_pil(pdf_path, page_index=args.page_index, dpi=300)
    draw = ImageDraw.Draw(image)

    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=args.threshold, target_sizes=target_sizes
    )[0]

    txt_lines = []
    txt_lines.append(f"Threshold used: {args.threshold}")
    txt_lines.append(f"Detections found: {len(results['scores'])}")
    txt_lines.append("")

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        txt_lines.append(f"{label_name}\t{float(score):.4f}\t{box}")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label_name} {float(score):.2f}", fill="red")

    if len(results["scores"]) == 0:
        txt_lines.append("No detections above threshold.")

    stem = f"{pdf_path.stem}_p{args.page_index+1}"
    img_out = out_dir / f"{stem}_detected.png"
    txt_out = out_dir / f"{stem}_detections.txt"

    image.save(img_out)
    txt_out.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Saved image: {img_out}")
    print(f"Saved detections: {txt_out}")
    print(f"Detections found: {len(results['scores'])}")


if __name__ == "__main__":
    main()
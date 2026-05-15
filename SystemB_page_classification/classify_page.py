#!/usr/bin/env python3
"""
classify_page.py
Location: ~/digitize_medical_records/SystemB_page_classification/classify_page.py

Visual page classifier using Qwen2.5-VL 7B.
Asks the model one simple yes/no question per page:
  "Does this page have radio button groups (CNR form)?"

Much more accurate than OpenCV circle detection.
~5-10 seconds per page with 7B model.

Usage:
  python classify_page.py /path/to/page.png
  python classify_page.py /path/to/page.png --debug
"""

import sys
import argparse
import tempfile
import os
from pathlib import Path

CLASSIFY_MODEL = "qwen2.5vl:7b"

CLASSIFY_PROMPT = """Look at this scanned medical document page.

Does this page contain radio button groups? Radio buttons are small printed circles (like ○ or ●) arranged in rows with text labels next to them such as "Oui", "Non", "NSP", "Masculin", "Féminin", "Urbain", "Rural", or similar options.

A CNR Paludisme form has MANY rows of these radio button groups (typically 10 or more per page).

A clinical report, lab table, or hospital letter does NOT have radio button groups - it only has plain text, tables with numbers, or checkboxes on the far right edge.

Answer with ONLY one word: YES or NO

YES = this page has radio button groups (CNR form)
NO = this page has no radio button groups (clinical report, lab table, letter, etc.)"""


def to_jpeg(image_path: str) -> str:
    """Convert PNG to JPEG - fixes Qwen PNG crash bug."""
    if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
        return image_path
    try:
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img = Image.open(image_path).convert("RGB")
        img.save(tmp.name, "JPEG", quality=85)
        return tmp.name
    except ImportError:
        return image_path  # try anyway


def classify_with_qwen(image_path: str, model: str = CLASSIFY_MODEL) -> dict:
    """
    Ask Qwen 7B to classify the page.
    Returns dict with is_cnr, confidence, answer, details.
    """
    try:
        import ollama
        import os as _os
    except ImportError:
        print("  ERROR: ollama not installed. Run: pip install ollama")
        return {"is_cnr": False, "confidence": 0.0,
                "answer": "ERROR", "details": "ollama not installed"}

    # Convert to JPEG if needed
    img_path = to_jpeg(image_path)
    tmp_created = img_path != image_path

    try:
        _ollama_host = _os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        _client = ollama.Client(host=_ollama_host)
        response = _client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": CLASSIFY_PROMPT,
                "images": [img_path]
            }]
        )
        answer = response["message"]["content"].strip().upper()

        # Parse answer - look for YES or NO anywhere in response
        if "YES" in answer:
            is_cnr = True
            confidence = 0.95
        elif "NO" in answer:
            is_cnr = False
            confidence = 0.95
        else:
            # Ambiguous - default to non-CNR (safer)
            is_cnr = False
            confidence = 0.5
            answer = f"AMBIGUOUS: {answer[:50]}"

        return {
            "is_cnr": is_cnr,
            "confidence": confidence,
            "answer": answer,
            "details": f"Qwen7B: {answer[:30]}"
        }

    except Exception as e:
        return {
            "is_cnr": False,
            "confidence": 0.0,
            "answer": f"ERROR: {e}",
            "details": f"Error: {e}"
        }
    finally:
        if tmp_created and os.path.exists(img_path):
            os.unlink(img_path)


def is_cnr_form(image_path: str,
                min_radio_groups: int = 2,   # kept for API compatibility
                min_circles_per_group: int = 2,
                debug: bool = False) -> dict:
    """
    Main classifier function - uses Qwen 7B vision model.
    min_radio_groups kept for API compatibility with process_patient.py.
    """
    result = classify_with_qwen(image_path, model=CLASSIFY_MODEL)

    if debug:
        # Save a simple debug marker (no OpenCV needed)
        try:
            import cv2
            import numpy as np
            image = cv2.imread(image_path)
            if image is not None:
                label = f"Qwen: {'CNR' if result['is_cnr'] else 'non-CNR'} | {result['answer'][:40]}"
                color = (0, 255, 0) if result["is_cnr"] else (0, 0, 255)
                cv2.putText(image, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                debug_dir  = Path(image_path).parent / "debug"
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / (Path(image_path).stem + "_classified.jpg")
                cv2.imwrite(str(debug_path), image)
                print(f"  Debug → {debug_path}")
        except Exception:
            pass  # debug is optional

    return {
        "is_cnr": result["is_cnr"],
        "confidence": result["confidence"],
        "circles_found": 0,
        "radio_groups_found": 0,
        "details": result["details"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify if a page is a CNR form using Qwen 7B vision"
    )
    parser.add_argument("image_path", help="Path to page image")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", default=CLASSIFY_MODEL)
    parser.add_argument("--min-groups",  type=int, default=2)
    parser.add_argument("--min-circles", type=int, default=2)
    args = parser.parse_args()

    result = is_cnr_form(args.image_path, debug=args.debug)

    print(f"\n{'='*55}")
    print(f"  File           : {args.image_path}")
    print(f"  Is CNR form    : {'✅ YES' if result['is_cnr'] else '❌ NO'}")
    print(f"  Confidence     : {result['confidence']:.0%}")
    print(f"  Details        : {result['details']}")
    print(f"{'='*55}\n")
    sys.exit(0 if result["is_cnr"] else 1)


if __name__ == "__main__":
    main()
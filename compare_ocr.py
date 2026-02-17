"""
OCR Engine Comparison: PaddleOCR vs Tesseract
Outputs side-by-side quality metrics and text comparison.

Usage:
    python compare_ocr.py yourfile.pdf
    python compare_ocr.py yourfile.pdf --pages 3
    python compare_ocr.py yourfile.pdf --save
"""
import sys
import os
import time
import argparse
import pdfplumber
from pathlib import Path

# Suppress PaddleOCR noise
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from ocr_engine import OCREngine


def score_text(text: str) -> dict:
    """
    Compute basic quality metrics for OCR output.
    - char_count: raw character count
    - word_count: number of words
    - alpha_ratio: proportion of alphabetic chars (higher = cleaner)
    - unique_words: vocabulary size (low = repetitive/garbage)
    """
    if not text:
        return {'char_count': 0, 'word_count': 0, 'alpha_ratio': 0.0, 'unique_words': 0}

    words = text.split()
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text.replace(' ', '').replace('\n', ''))

    return {
        'char_count': len(text),
        'word_count': len(words),
        'alpha_ratio': round(alpha_chars / total_chars, 3) if total_chars else 0,
        'unique_words': len(set(w.lower() for w in words))
    }


def compare_page(page, page_num: int, paddle_engine: OCREngine, tess_engine: OCREngine) -> dict:
    """Run both engines on one page and return comparison results."""
    img = page.to_image(resolution=300).original

    # PaddleOCR
    t0 = time.time()
    paddle_text = paddle_engine.extract_text_paddle(paddle_engine.preprocess_image(img)) or ""
    paddle_time = round(time.time() - t0, 2)

    # Tesseract
    t0 = time.time()
    tess_text = tess_engine.extract_text_tesseract(tess_engine.preprocess_image(img)) or ""
    tess_time = round(time.time() - t0, 2)

    return {
        'page': page_num,
        'paddle': {'text': paddle_text, 'time': paddle_time, 'score': score_text(paddle_text)},
        'tesseract': {'text': tess_text, 'time': tess_time, 'score': score_text(tess_text)},
    }


def print_report(results: list):
    """Print formatted comparison report."""
    W = 60  # column width

    print(f"\n{'='*130}")
    print(f"{'OCR COMPARISON REPORT':^130}")
    print(f"{'='*130}")
    print(f"{'METRIC':<25} {'PADDLEOCR':>{W}} {'TESSERACT':>{W}}")
    print(f"{'-'*130}")

    total_paddle_chars = 0
    total_tess_chars = 0
    total_paddle_time = 0
    total_tess_time = 0

    for r in results:
        psc = r['paddle']['score']
        tsc = r['tesseract']['score']
        total_paddle_chars += psc['char_count']
        total_tess_chars += tsc['char_count']
        total_paddle_time += r['paddle']['time']
        total_tess_time += r['tesseract']['time']

        print(f"\n  PAGE {r['page']}")
        print(f"  {'Chars extracted':<23} {psc['char_count']:>{W}} {tsc['char_count']:>{W}}")
        print(f"  {'Words':<23} {psc['word_count']:>{W}} {tsc['word_count']:>{W}}")
        print(f"  {'Alpha ratio':<23} {psc['alpha_ratio']:>{W}} {tsc['alpha_ratio']:>{W}}")
        print(f"  {'Unique words':<23} {psc['unique_words']:>{W}} {tsc['unique_words']:>{W}}")
        print(f"  {'Time (sec)':<23} {r['paddle']['time']:>{W}} {r['tesseract']['time']:>{W}}")

    print(f"\n{'='*130}")
    print(f"  TOTAL")
    print(f"  {'Total chars':<23} {total_paddle_chars:>{W}} {total_tess_chars:>{W}}")
    print(f"  {'Total time (sec)':<23} {round(total_paddle_time,2):>{W}} {round(total_tess_time,2):>{W}}")
    print(f"{'='*130}\n")

    winner = "PaddleOCR" if total_paddle_chars >= total_tess_chars else "Tesseract"
    print(f"  → More text extracted by: {winner}")
    print()


def save_results(results: list, pdf_path: str):
    """Save each engine's output to separate text files."""
    stem = Path(pdf_path).stem
    paddle_out = f"{stem}_paddle.txt"
    tess_out = f"{stem}_tesseract.txt"

    with open(paddle_out, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"\n--- Page {r['page']} ---\n{r['paddle']['text']}\n")

    with open(tess_out, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"\n--- Page {r['page']} ---\n{r['tesseract']['text']}\n")

    print(f"  Saved: {paddle_out}")
    print(f"  Saved: {tess_out}")


def main():
    parser = argparse.ArgumentParser(description='Compare PaddleOCR vs Tesseract on a PDF')
    parser.add_argument('pdf', help='Path to PDF file')
    parser.add_argument('--pages', type=int, default=3, help='Number of pages to test (default: 3)')
    parser.add_argument('--save', action='store_true', help='Save output text files for manual review')
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: File not found: {args.pdf}")
        sys.exit(1)

    # Init both engines independently (not as fallback)
    print("Initializing PaddleOCR...")
    paddle_engine = OCREngine(use_paddle=True, use_tesseract_fallback=False)

    print("Initializing Tesseract...")
    tess_engine = OCREngine(use_paddle=False, use_tesseract_fallback=True)

    if not paddle_engine.use_paddle:
        print("ERROR: PaddleOCR failed to initialize.")
        sys.exit(1)
    if not tess_engine.use_tesseract_fallback:
        print("ERROR: Tesseract failed to initialize. See Tesseract install steps below.")
        sys.exit(1)

    # Process pages
    results = []
    print(f"\nProcessing {args.pdf} ({args.pages} pages)...\n")

    with pdfplumber.open(args.pdf) as pdf:
        pages_to_test = min(args.pages, len(pdf.pages))
        for page_num, page in enumerate(pdf.pages[:pages_to_test], 1):
            print(f"  Page {page_num}/{pages_to_test}...", end='\r')
            results.append(compare_page(page, page_num, paddle_engine, tess_engine))

    # Report
    print_report(results)

    # Preview first page text from each engine
    if results:
        print("  PADDLE TEXT PREVIEW (Page 1, first 300 chars):")
        print(f"  {'-'*80}")
        print(f"  {results[0]['paddle']['text'][:300].replace(chr(10), chr(10)+'  ')}")
        print()
        print("  TESSERACT TEXT PREVIEW (Page 1, first 300 chars):")
        print(f"  {'-'*80}")
        print(f"  {results[0]['tesseract']['text'][:300].replace(chr(10), chr(10)+'  ')}")
        print()

    if args.save:
        save_results(results, args.pdf)


if __name__ == '__main__':
    main()

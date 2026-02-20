"""
OCR module — extract text from scanned PDFs using Docling + EasyOCR (French).

Does not go recursively through subfolders, but can process multiple PDFs in a single folder. 
To process subfolders recursively, run pipeline.py

Usage standalone:
    python path/ocr.py /path/to/pdf_folder /path/to/output
    python path/ocr.py /path/to/single_file.pdf /path/to/output --gpu

Requirements:
    pip install docling easyocr
"""

import logging
import warnings
from pathlib import Path

# Suppress known deprecation warnings from Docling internals
warnings.filterwarnings("ignore", message=".*Deprecated field.*use_gpu.*")
warnings.filterwarnings("ignore", message=".*strict_text.*deprecated.*")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)

# Suppress noisy internal logs
for _name in ("docling", "RapidOCR", "easyocr", "PIL", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _build_converter(
    use_gpu: bool = False,
    num_threads: int = 4,
    artifacts_path: str = None,
) -> DocumentConverter:
    """
    Build a Docling converter optimised for French scanned medical PDFs.

    - EasyOCR with lang=["fr"] (not the default RapidOCR which uses Chinese models)
    - force_full_page_ocr=True — all pages are scanned images
    - do_table_structure=True — preserves layout of lab results
    - GPU via CUDA when available on server

    For air-gapped use:
        Pre-download:  docling-tools models download -o /path/to/models
        Then pass:     artifacts_path="/path/to/models"
        EasyOCR cache: ~/.EasyOCR/model/ (copy from internet machine)
    """
    ocr_options = EasyOcrOptions(
        lang=["fr"],
        force_full_page_ocr=True,
        use_gpu=False,  # deprecated field — GPU is controlled by accelerator_options below
    )

    if use_gpu:
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA, num_threads=num_threads)
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=num_threads)

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_options,
        accelerator_options=accel,
        artifacts_path=artifacts_path,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# Module-level converter — initialized lazily on first use
_converter = None


def _get_converter(
    use_gpu: bool = False,
    num_threads: int = 4,
    artifacts_path: str = None,
) -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = _build_converter(
            use_gpu=use_gpu,
            num_threads=num_threads,
            artifacts_path=artifacts_path,
        )
    return _converter


def pdf_to_text(
    pdf_path: Path,
    use_gpu: bool = False,
    artifacts_path: str = None,
) -> str:
    """Extract text from a single PDF."""
    converter = _get_converter(use_gpu=use_gpu, artifacts_path=artifacts_path)
    result = converter.convert(str(pdf_path))
    return result.document.export_to_text()


def process_folder(
    folder: Path,
    use_gpu: bool = False,
    artifacts_path: str = None,
    extensions: tuple = (".pdf", ".PDF"),
) -> dict[str, str]:
    """
    OCR all PDFs in a folder.
    Returns: {filename: extracted_text}
    """
    pdf_files = sorted(
        f for f in folder.iterdir() if f.suffix in extensions and f.is_file()
    )
    results = {}
    converter = _get_converter(use_gpu=use_gpu, artifacts_path=artifacts_path)
    for pdf in pdf_files:
        logger.info(f"OCR: {pdf.name}")
        try:
            result = converter.convert(str(pdf))
            text = result.document.export_to_text()
            results[pdf.name] = text
            logger.info(f"  → {len(text)} chars extracted")
        except Exception as e:
            logger.error(f"  OCR failed for {pdf.name}: {e}")
            results[pdf.name] = ""
    return results


# --- CLI entry point ---
if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="OCR French medical PDFs via Docling + EasyOCR"
    )
    parser.add_argument("input", help="PDF file or folder of PDFs")
    parser.add_argument("output", help="Output folder for .txt files")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (CUDA)")
    parser.add_argument(
        "--artifacts-path",
        default=None,
        help="Path to pre-downloaded Docling models (for offline use)",
    )
    args = parser.parse_args()

    target = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        logger.info(f"Processing: {target.name} (GPU={args.gpu})")
        text = pdf_to_text(target, use_gpu=args.gpu, artifacts_path=args.artifacts_path)
        dest = out_dir / (target.stem + ".txt")
        dest.write_text(text, encoding="utf-8")
        logger.info(f"  → {dest} ({len(text)} chars)")

    elif target.is_dir():
        results = process_folder(
            target, use_gpu=args.gpu, artifacts_path=args.artifacts_path
        )
        for name, text in results.items():
            dest = out_dir / (Path(name).stem + ".txt")
            dest.write_text(text, encoding="utf-8")
        logger.info(f"Done. {len(results)} files → {out_dir}")

    else:
        logger.error(f"Not found: {target}")
        sys.exit(1)
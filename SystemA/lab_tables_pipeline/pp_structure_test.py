from pathlib import Path
import argparse
from paddlex import create_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run PP-StructureV3 on one PDF")
    parser.add_argument("input_path", help="Full path to the PDF file")
    parser.add_argument(
        "--output-dir",
        default="/home/lfarooq/digitize_medical_records/benchmark_outputs/pp_structure",
        help="Directory where PP-StructureV3 outputs will be saved",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = create_pipeline(pipeline="PP-StructureV3")

    results = pipeline.predict(
        input=str(input_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    for res in results:
        try:
            res.print()
        except Exception:
            pass

        try:
            res.save_to_json(str(output_dir))
        except Exception as e:
            print(f"Could not save JSON: {e}")

        try:
            res.save_to_markdown(str(output_dir))
        except Exception as e:
            print(f"Could not save Markdown: {e}")

        try:
            res.save_to_html(str(output_dir))
        except Exception as e:
            print(f"Could not save HTML: {e}")

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
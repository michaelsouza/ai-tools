import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from tools.pdf2md import (
    build_mistral_ocr_payload,
    extract_pages_content_and_save_images_mistral,
    generate_ocr_json_filename,
    save_ocr_response_to_file,
)


class Pdf2MdMistralTests(unittest.TestCase):
    def test_builds_ocr4_payload_with_optional_fields(self):
        args = SimpleNamespace(
            mistral_model="mistral-ocr-4-0",
            include_images=True,
            table_format="markdown",
            extract_header=True,
            extract_footer=False,
            include_blocks=True,
            confidence_scores="page",
        )

        payload = build_mistral_ocr_payload("https://signed.example/file.pdf", args)

        self.assertEqual(payload["model"], "mistral-ocr-4-0")
        self.assertEqual(
            payload["document"],
            {"type": "document_url", "document_url": "https://signed.example/file.pdf"},
        )
        self.assertTrue(payload["include_image_base64"])
        self.assertEqual(payload["table_format"], "markdown")
        self.assertTrue(payload["extract_header"])
        self.assertTrue(payload["include_blocks"])
        self.assertEqual(payload["confidence_scores_granularity"], "page")
        self.assertNotIn("extract_footer", payload)

    def test_extracts_markdown_from_raw_json_response_and_strips_unsaved_images(self):
        response = {
            "pages": [
                {
                    "index": 0,
                    "markdown": "Intro\n\n![img-0.jpeg](img-0.jpeg)\n\nText",
                    "images": [{"id": "img-0.jpeg"}],
                }
            ]
        }
        console = Console(file=io.StringIO())

        parts = extract_pages_content_and_save_images_mistral(
            response,
            include_image_base64=False,
            console=console,
            images_dir=None,
            output_dir=None,
            pdf_stem="sample",
        )

        self.assertEqual(parts, ["Intro\n\nText"])

    def test_saves_sidecar_json_next_to_markdown_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "sample.md"
            json_path = generate_ocr_json_filename(str(md_path))

            save_ocr_response_to_file({"pages": [{"markdown": "Hello"}]}, json_path)

            self.assertEqual(Path(json_path).name, "sample.ocr.json")
            self.assertIn('"markdown": "Hello"', Path(json_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

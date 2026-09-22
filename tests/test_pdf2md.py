import base64
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib import error as urlerror

import PyPDF2
from PIL import Image
from rich.console import Console

from tools.pdf2md import (
    MISTRAL_MAX_BACKOFF_SECONDS,
    MistralWorkspaceBlocked,
    LIGHTON_MAX_SIDE,
    LIGHTON_MAX_TOKENS,
    build_lighton_payload,
    build_mistral_ocr_payload,
    delete_mistral_file,
    extract_pages_content_and_save_images_mistral,
    fit_longest_side,
    generate_ocr_json_filename,
    is_mistral_workspace_blocked,
    mistral_retry_delay,
    process_lighton_ocr,
    process_ocr_with_mistral,
    process_single_pdf,
    save_ocr_response_to_file,
    upload_pdf_to_mistral,
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


def _http_error(code, headers=None, body=b'{"message": "Rate limit exceeded"}'):
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urlerror.HTTPError("https://api.mistral.ai/v1/ocr", code, "error", message, io.BytesIO(body))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Pdf2MdMistralRateLimitTests(unittest.TestCase):
    ARGS = SimpleNamespace(
        mistral_model="mistral-ocr-latest",
        include_images=False,
        table_format="none",
        extract_header=False,
        extract_footer=False,
        include_blocks=False,
        confidence_scores=None,
    )

    def test_zero_limit_means_blocked_workspace_case_insensitive(self):
        self.assertTrue(is_mistral_workspace_blocked(429, {"X-RateLimit-Limit-Req-Minute": "0"}))
        self.assertFalse(is_mistral_workspace_blocked(429, {"x-ratelimit-limit-req-minute": "60"}))
        self.assertFalse(is_mistral_workspace_blocked(429, {}))
        self.assertFalse(is_mistral_workspace_blocked(500, {"x-ratelimit-limit-req-minute": "0"}))

    def test_retry_delay_honours_retry_after_and_caps(self):
        self.assertEqual(mistral_retry_delay(429, {"Retry-After": "7"}, attempt=1), 7)
        self.assertEqual(mistral_retry_delay(429, {}, attempt=3), 8)
        self.assertEqual(mistral_retry_delay(503, {"Retry-After": "999"}, attempt=1), MISTRAL_MAX_BACKOFF_SECONDS)
        self.assertEqual(mistral_retry_delay(503, {}, attempt=10), MISTRAL_MAX_BACKOFF_SECONDS)
        self.assertIsNone(mistral_retry_delay(400, {}, attempt=1))
        self.assertIsNone(mistral_retry_delay(401, {}, attempt=1))

    def _run(self, side_effect):
        console = Console(file=io.StringIO())
        with mock.patch.dict("os.environ", {"MISTRAL_API_KEY": "k"}), mock.patch(
            "tools.pdf2md.urlrequest.urlopen", side_effect=side_effect
        ) as urlopen, mock.patch("tools.pdf2md.time.sleep") as sleep:
            result = process_ocr_with_mistral("https://signed.example/f.pdf", self.ARGS, console)
        return result, urlopen, sleep

    def test_blocked_workspace_raises_without_retrying(self):
        with self.assertRaises(MistralWorkspaceBlocked):
            self._run([_http_error(429, {"x-ratelimit-limit-req-minute": "0"})])

    def test_transient_429_is_retried_then_succeeds(self):
        result, urlopen, sleep = self._run(
            [_http_error(429, {"x-ratelimit-limit-req-minute": "60", "Retry-After": "3"}), _Response(b'{"pages": []}')]
        )
        self.assertEqual(result, {"pages": []})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(3.0)

    def test_client_error_is_not_retried(self):
        result, urlopen, sleep = self._run([_http_error(400, body=b'{"message": "bad"}')])
        self.assertIsNone(result)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_gives_up_after_max_attempts(self):
        result, urlopen, sleep = self._run(lambda *a, **k: (_ for _ in ()).throw(_http_error(503)))
        self.assertIsNone(result)
        self.assertEqual(urlopen.call_count, 5)
        self.assertEqual(sleep.call_count, 4)


class _FakeFiles:
    def __init__(self, fail_signed_url=False, fail_delete=False):
        self.fail_signed_url = fail_signed_url
        self.fail_delete = fail_delete
        self.deleted = []

    def upload(self, file, purpose):
        return SimpleNamespace(id="file-1")

    def get_signed_url(self, file_id, expiry):
        if self.fail_signed_url:
            raise RuntimeError("signed url failed")
        return SimpleNamespace(url="https://signed.example/f.pdf")

    def delete(self, file_id):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted.append(file_id)


class Pdf2MdMistralUploadCleanupTests(unittest.TestCase):
    PDF = Path(__file__).parent / "fixtures" / "lavor2019polynomiality.pdf"

    def setUp(self):
        self.console = Console(file=io.StringIO())

    def _args(self):
        return SimpleNamespace(
            model="mistral",
            mistral_model="mistral-ocr-latest",
            pages=None,
            yes=True,
            include_images=False,
            save_ocr_json=False,
        )

    def test_upload_returns_url_and_id_and_keeps_file(self):
        client = SimpleNamespace(files=_FakeFiles())
        result = upload_pdf_to_mistral(client, str(self.PDF), self.console)
        self.assertEqual(result, ("https://signed.example/f.pdf", "file-1"))
        self.assertEqual(client.files.deleted, [])

    def test_upload_deletes_file_when_signed_url_fails(self):
        client = SimpleNamespace(files=_FakeFiles(fail_signed_url=True))
        self.assertIsNone(upload_pdf_to_mistral(client, str(self.PDF), self.console))
        self.assertEqual(client.files.deleted, ["file-1"])

    def test_delete_failure_only_warns(self):
        client = SimpleNamespace(files=_FakeFiles(fail_delete=True))
        self.assertFalse(delete_mistral_file(client, "file-1", self.console))
        self.assertIn("could not delete", self.console.file.getvalue())

    def test_single_pdf_deletes_upload_after_successful_ocr(self):
        client = SimpleNamespace(files=_FakeFiles())
        response = {"pages": [{"index": 0, "markdown": "Hello"}]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "tools.pdf2md.process_ocr_with_mistral", return_value=response
        ):
            output = Path(tmp) / "out.md"
            ok = process_single_pdf(str(self.PDF), str(output), self.console, self._args(), client, False)
            self.assertTrue(ok)
            self.assertIn("Hello", output.read_text(encoding="utf-8"))
        self.assertEqual(client.files.deleted, ["file-1"])

    def test_single_pdf_deletes_upload_when_workspace_blocked(self):
        client = SimpleNamespace(files=_FakeFiles())
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "tools.pdf2md.process_ocr_with_mistral", side_effect=MistralWorkspaceBlocked("limit 0")
        ):
            with self.assertRaises(MistralWorkspaceBlocked):
                process_single_pdf(str(self.PDF), str(Path(tmp) / "out.md"), self.console, self._args(), client, False)
        self.assertEqual(client.files.deleted, ["file-1"])


class Pdf2MdLightOnTests(unittest.TestCase):
    PDF = Path(__file__).parent / "fixtures" / "lavor2019polynomiality.pdf"

    def _response(self, content, finish="stop"):
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        return _Response(json.dumps(body).encode())

    def _run(self, responses):
        console = Console(file=io.StringIO())
        args = SimpleNamespace(server_url="http://127.0.0.1:9/v1")
        with mock.patch("tools.pdf2md.urlrequest.urlopen", side_effect=responses) as urlopen:
            parts = process_lighton_ocr(str(self.PDF), console, args)
        return parts, urlopen, console.file.getvalue()

    def test_fit_longest_side_downscales_only(self):
        big = Image.new("RGB", (1700, 2200))
        small = Image.new("RGB", (800, 1000))
        self.assertEqual(max(fit_longest_side(big, LIGHTON_MAX_SIDE).size), LIGHTON_MAX_SIDE)
        self.assertEqual(fit_longest_side(big, LIGHTON_MAX_SIDE).size, (1190, 1540))
        self.assertIs(fit_longest_side(small, LIGHTON_MAX_SIDE), small)

    def test_payload_is_image_only_with_model_card_sampling(self):
        payload = build_lighton_payload("QUJD")
        content = payload["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["image_url"])
        self.assertEqual(content[0]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual((payload["temperature"], payload["top_p"], payload["max_tokens"]), (0.2, 0.9, LIGHTON_MAX_TOKENS))

    def test_one_request_per_page_with_downscaled_image(self):
        pages = len(PyPDF2.PdfReader(str(self.PDF)).pages)
        parts, urlopen, _ = self._run([self._response(f"page {i}") for i in range(pages)])
        self.assertEqual(parts, [f"page {i}" for i in range(pages)])
        self.assertEqual(urlopen.call_count, pages)
        request = urlopen.call_args_list[0].args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9/v1/chat/completions")
        image_b64 = json.loads(request.data)["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1]
        self.assertLessEqual(max(Image.open(io.BytesIO(base64.b64decode(image_b64))).size), LIGHTON_MAX_SIDE)

    def test_token_limit_warns_about_degenerate_output(self):
        pages = len(PyPDF2.PdfReader(str(self.PDF)).pages)
        responses = [self._response("x", finish="length")] + [self._response("ok") for _ in range(pages - 1)]
        parts, _, out = self._run(responses)
        self.assertEqual(len(parts), pages)
        self.assertIn("token limit", out)
        self.assertIn("Q8_0", out)

    def test_server_error_aborts_the_pdf(self):
        parts, urlopen, out = self._run(urlerror.URLError("connection refused"))
        self.assertIsNone(parts)
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("LightOnOCR-2 error on page 1", out)


if __name__ == "__main__":
    unittest.main()

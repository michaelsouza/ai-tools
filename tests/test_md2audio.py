import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from tools import md2audio
from tools.md2audio import chunk_blocks, markdown_to_blocks

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class MarkdownToAudioTests(unittest.TestCase):
    def test_renders_headings_lists_and_links_as_prose(self):
        markdown = """# Title

Read the [paper](https://example.com).

- First item
- Second item
"""

        blocks = markdown_to_blocks(markdown, "en")

        self.assertEqual(blocks, ["Title", "Read the paper.", "First item", "Second item"])

    def test_uses_omission_cues_for_non_prose_blocks(self):
        markdown = """Intro.

```python
print("hello")
```

| A | B |
|---|---|
| 1 | 2 |

<div>raw</div>

![Plot](plot.png)

[^1]: Extra note.
"""

        blocks = markdown_to_blocks(markdown, "en")

        self.assertIn("Code block omitted.", blocks)
        self.assertIn("Table omitted.", blocks)
        self.assertIn("HTML block omitted.", blocks)
        self.assertIn("Image omitted.", blocks)
        self.assertIn("Footnote omitted.", blocks)

    def test_portuguese_omission_cues(self):
        blocks = markdown_to_blocks("```text\nx\n```\n\n[^1]: nota", "pt-br")

        self.assertEqual(blocks, ["Bloco de codigo omitido.", "Nota de rodape omitida."])

    def test_chunks_on_block_boundaries(self):
        chunks = chunk_blocks(["A short paragraph.", "Another short paragraph.", "Final paragraph."], 43)

        self.assertEqual(chunks, ["A short paragraph.", "Another short paragraph.\n\nFinal paragraph."])

    def test_sample_fixture_renders_to_narration_chunks(self):
        markdown = (FIXTURES_DIR / "sample.md").read_text(encoding="utf-8")

        blocks = markdown_to_blocks(markdown, "en")
        chunks = chunk_blocks(blocks, 900)

        self.assertGreater(len(blocks), 10)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(blocks[0], "Remarks on Magnifica Humanitas")
        self.assertIn("What these systems are", blocks)
        self.assertTrue(all("|" not in chunk for chunk in chunks))


class FakeResponse:
    def __init__(self, status_code, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.ok = 200 <= status_code < 300


class GeminiEngineTests(unittest.TestCase):
    def test_request_asks_openrouter_for_pcm_with_model_and_voice(self):
        with mock.patch("requests.post", return_value=FakeResponse(200, b"\x01\x00" * 10)) as post:
            pcm = md2audio.gemini_speech("Olá.", md2audio.GEMINI_MODEL, "Kore", "key")

        self.assertEqual(pcm, b"\x01\x00" * 10)
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], md2audio.OPENROUTER_SPEECH_URL)
        self.assertEqual(
            kwargs["json"],
            {"model": md2audio.GEMINI_MODEL, "input": "Olá.", "voice": "Kore", "response_format": "pcm"},
        )
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer key"})

    def test_retries_rate_limit_then_succeeds(self):
        responses = [FakeResponse(429, text="slow down"), FakeResponse(200, b"\x00\x00")]
        with mock.patch("requests.post", side_effect=responses) as post, mock.patch("time.sleep"):
            self.assertEqual(md2audio.gemini_speech("x", "m", "v", "k"), b"\x00\x00")
        self.assertEqual(post.call_count, 2)

    def test_client_error_is_not_retried(self):
        with mock.patch("requests.post", return_value=FakeResponse(400, text="bad voice")) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                md2audio.gemini_speech("x", "m", "v", "k")
        self.assertEqual(post.call_count, 1)

    def test_chunks_become_24khz_mono_wavs_in_order(self):
        pcm = {"Um.": b"\x01\x00" * 24000, "Dois.": b"\x02\x00" * 12000}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            with mock.patch.object(md2audio, "gemini_speech", side_effect=lambda text, *_: pcm[text]):
                paths, _, requests_made = md2audio.synthesize_gemini(
                    ["Um.", "Dois."], Path(tmp), "m", "Kore", 2, lambda _: None
                )
            output = Path(tmp) / "out.wav"
            md2audio.concatenate_wavs(paths, output)
            with wave.open(str(output), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))
                self.assertEqual(wav.getnframes(), 36000)
                self.assertEqual(wav.readframes(1), b"\x01\x00")
        self.assertEqual(requests_made, 2)

    def test_missing_api_key_points_to_local_engine(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "--engine qwen"):
                md2audio.synthesize_gemini(["x"], Path("."), "m", "v", 1, lambda _: None)

    def test_instruct_is_rejected_for_gemini(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "a.md"
            doc.write_text("Olá.", encoding="utf-8")
            argv = ["md2audio.py", str(doc), "--language", "pt-br", "--instruct", "calmo"]
            with mock.patch("sys.argv", argv):
                self.assertEqual(md2audio.main(), 2)


class OutputLocationTests(unittest.TestCase):
    def test_default_output_goes_to_repo_outputs_audio(self):
        path = md2audio.default_output_path(Path("/anywhere/joao_1-5.md"))

        self.assertEqual(path, md2audio.REPO_ROOT / "outputs" / "audio" / "joao_1-5.wav")

    def test_source_markdown_is_copied_beside_the_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in" / "texto.md"
            source.parent.mkdir()
            source.write_text("Olá.", encoding="utf-8")
            output = Path(tmp) / "out" / "texto.wav"
            output.parent.mkdir()

            copy = md2audio.keep_source_next_to_audio(source, output)

            self.assertEqual(copy, Path(tmp) / "out" / "texto.md")
            self.assertEqual(copy.read_text(encoding="utf-8"), "Olá.")

    def test_source_already_beside_the_audio_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "texto.md"
            source.write_text("Olá.", encoding="utf-8")

            self.assertIsNone(md2audio.keep_source_next_to_audio(source, Path(tmp) / "texto.wav"))


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from tools import md2audio, voice_ref
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
            argv = ["md2audio.py", str(doc), "--engine", "gemini", "--instruct", "calmo"]
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


class VoiceCloneTests(unittest.TestCase):
    def run_main(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "a.md"
            doc.write_text("Olá.", encoding="utf-8")
            with mock.patch("sys.argv", ["md2audio.py", str(doc), "--language", "pt-br", *extra]):
                return md2audio.main()

    def test_missing_voice_points_to_voice_ref(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md2audio, "VOICES_DIR", Path(tmp)):
            with self.assertRaisesRegex(md2audio.MarkdownNarrationError, "voice_ref.py ninguem"):
                md2audio.load_voice("ninguem")

    def test_voice_loads_reference_audio_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md2audio, "VOICES_DIR", Path(tmp)):
            (Path(tmp) / "eu.wav").write_bytes(b"RIFF")
            (Path(tmp) / "eu.txt").write_text("Bom dia!\n", encoding="utf-8")

            self.assertEqual(md2audio.load_voice("eu"), (Path(tmp) / "eu.wav", "Bom dia!"))

    def test_voice_selects_local_clone_model(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md2audio, "VOICES_DIR", Path(tmp)):
            (Path(tmp) / "eu.wav").write_bytes(b"RIFF")
            (Path(tmp) / "eu.txt").write_text("Bom dia!", encoding="utf-8")
            with mock.patch.object(md2audio, "synthesize_chunks", side_effect=RuntimeError("stop")) as synth:
                self.assertEqual(self.run_main("--voice", "eu"), 1)

        kwargs = synth.call_args.kwargs
        self.assertEqual(kwargs["engine"], "qwen")
        self.assertEqual(kwargs["model_name_or_path"], md2audio.CLONE_MODEL)
        self.assertEqual(kwargs["voice"], (Path(tmp) / "eu.wav", "Bom dia!"))

    def test_voice_rejects_gemini_speaker_and_instruct(self):
        for extra in (["--engine", "gemini"], ["--speaker", "ryan"], ["--instruct", "calmo"]):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_main("--voice", "eu", *extra), 2)


class VoiceReferenceTests(unittest.TestCase):
    def test_reading_texts_come_from_the_doc(self):
        texts = voice_ref.load_texts()

        self.assertEqual(sorted(texts), ["A", "B", "C"])
        self.assertTrue(texts["A"].startswith("Bom dia!"))

    def test_word_error_rate_ignores_case_and_punctuation(self):
        self.assertEqual(voice_ref.word_error_rate("Bom dia! Tudo bem?", "bom dia, tudo bem"), 0.0)
        self.assertAlmostEqual(voice_ref.word_error_rate("um dois três quatro", "um dois quatro"), 0.25)
        self.assertEqual(voice_ref.word_error_rate("diminui em até quarenta por cento", "diminui em até 40%"), 0.0)

    def test_problem_explains_unusable_takes(self):
        take = voice_ref.Take(Path("a.m4a"), Path("a.wav"), 15.0, 40.0, "", "A", 0.0)
        self.assertEqual(voice_ref.problem(take), "")
        self.assertEqual(voice_ref.problem(voice_ref.Take(Path("a"), Path("a"), 3.0, 40.0, "", "A", 0.0)), "too short")
        self.assertEqual(
            voice_ref.problem(voice_ref.Take(Path("a"), Path("a"), 15.0, 40.0, "", "A", 0.2)),
            "reading differs from text",
        )


def write_silence(path, seconds, rate=24000):
    md2audio.write_pcm_wav(path, b"\x00\x00" * int(seconds * rate), rate)


class TempoTests(unittest.TestCase):
    def test_tempo_outside_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "a.md"
            doc.write_text("Olá.", encoding="utf-8")
            for tempo in ("0.4", "2.5"):
                with self.subTest(tempo=tempo), mock.patch(
                    "sys.argv", ["md2audio.py", str(doc), "--language", "pt-br", "--tempo", tempo]
                ):
                    self.assertEqual(md2audio.main(), 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_tempo_stretches_the_final_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = [Path(tmp) / "c1.wav", Path(tmp) / "c2.wav"]
            for chunk in chunks:
                write_silence(chunk, 1.0)
            output = Path(tmp) / "out" / "narration.wav"
            with mock.patch.object(md2audio, "synthesize_gemini", return_value=(chunks, 0.0, 2)):
                metrics = md2audio.synthesize_chunks(
                    ["a", "b"],
                    output,
                    "pt-br",
                    "gemini",
                    "m",
                    "Kore",
                    None,
                    "auto",
                    1,
                    1,
                    False,
                    False,
                    md2audio.Console(quiet=True),
                    tempo=0.8,
                )
            self.assertAlmostEqual(metrics.audio_seconds, 2.0 / 0.8, delta=0.05)
            self.assertEqual(sorted(p.name for p in output.parent.iterdir()), ["narration.wav"])

    def test_missing_ffmpeg_is_explained(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "apt install ffmpeg"):
                md2audio.apply_tempo(Path("a.wav"), Path("b.wav"), 0.9)


class DefaultsTests(unittest.TestCase):
    def resolved(self, *argv):
        args = md2audio.build_parser().parse_args(["a.md", *argv])
        md2audio.resolve_defaults(args)
        return args

    def test_no_options_clone_the_users_voice_slowed_down(self):
        args = self.resolved()

        self.assertEqual(
            (args.language, args.engine, args.voice, args.model, args.tempo),
            ("pt-br", "qwen", "michael", md2audio.CLONE_MODEL, 0.86),
        )

    def test_preset_speaker_drops_the_clone_and_its_tempo(self):
        for argv in (["--speaker", "ryan"], ["--instruct", "Calm."]):
            with self.subTest(argv=argv):
                args = self.resolved(*argv)
                self.assertEqual((args.voice, args.model, args.tempo), (None, md2audio.DEFAULT_MODEL, 1.0))

    def test_gemini_drops_the_clone_and_its_tempo(self):
        args = self.resolved("--engine", "gemini")

        self.assertEqual((args.voice, args.model, args.speaker, args.tempo), (None, md2audio.GEMINI_MODEL, "Kore", 1.0))

    def test_explicit_tempo_always_wins(self):
        self.assertEqual(self.resolved("--tempo", "1.0").tempo, 1.0)
        self.assertEqual(self.resolved("--engine", "gemini", "--tempo", "0.9").tempo, 0.9)

    def test_explicit_voice_conflicts_are_errors(self):
        for argv in (["--engine", "gemini"], ["--speaker", "ryan"], ["--instruct", "Calm."]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.resolved("--voice", "michael", *argv)


if __name__ == "__main__":
    unittest.main()

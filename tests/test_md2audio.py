import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

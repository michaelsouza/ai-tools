import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.merge_bib import bibtex_text, load_bib, main, merge_entries


class MergeBibTests(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "refs.bib"
            path.write_text(text, encoding="utf-8")
            entries, errors = load_bib(path)
        self.assertEqual(errors, [])
        return entries

    def test_skips_duplicate_by_doi(self):
        first = self.parse_text("""@article{one,
  title = {A Useful Paper},
  doi = {10.1000/ABC}
}
""")
        second = self.parse_text("""@article{two,
  title = {A Useful Paper, Expanded},
  doi = {https://doi.org/10.1000/abc}
}
""")

        result = merge_entries(first, second, ["key", "doi", "title"], "rename", "_2")

        self.assertEqual([entry["ID"] for entry in result.entries], ["one"])
        self.assertEqual(result.skipped_duplicates, 1)

    def test_skips_duplicate_by_normalized_title(self):
        first = self.parse_text("""@article{one,
  title = {A {Useful} Paper}
}
""")
        second = self.parse_text("""@inproceedings{two,
  title = {A Useful Paper}
}
""")

        result = merge_entries(first, second, ["title"], "rename", "_2")

        self.assertEqual([entry["ID"] for entry in result.entries], ["one"])
        self.assertEqual(result.skipped_duplicates, 1)

    def test_renames_key_conflict_with_different_content(self):
        first = self.parse_text("""@article{smith2020,
  title = {First Paper}
}
""")
        second = self.parse_text("""@article{smith2020,
  title = {Second Paper}
}
""")

        result = merge_entries(first, second, ["key", "doi", "title"], "rename", "_b")

        self.assertEqual([entry["ID"] for entry in result.entries], ["smith2020", "smith2020_b"])
        self.assertIn("@article{smith2020_b,", bibtex_text([result.entries[1]]))
        self.assertEqual(result.renamed_conflicts, 1)

    def test_default_output_is_second_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "project_a.bib"
            second = Path(tmpdir) / "project_b.bib"
            first.write_text(
                """@article{one,
  title = {First Paper}
}
""",
                encoding="utf-8",
            )
            second.write_text(
                """@article{two,
  title = {Second Paper}
}
""",
                encoding="utf-8",
            )

            exit_code = main([str(first), str(second), "--quiet"])

            self.assertEqual(exit_code, 0)
            merged = second.read_text(encoding="utf-8")
            self.assertIn("@article{one,", merged)
            self.assertIn("@article{two,", merged)

    def test_stdout_prints_added_entries_before_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "project_a.bib"
            second = Path(tmpdir) / "project_b.bib"
            first.write_text(
                """@article{one,
  title = {First Paper}
}
""",
                encoding="utf-8",
            )
            second.write_text(
                """@article{two,
  title = {Second Paper}
}
""",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main([str(first), str(second)])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("@article{one,", output)
            self.assertNotIn("@article{two,", output)
            self.assertIn("BibTeX Merge Summary", output)
            self.assertLess(output.index("@article{one,"), output.index("BibTeX Merge Summary"))


if __name__ == "__main__":
    unittest.main()

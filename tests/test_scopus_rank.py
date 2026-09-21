import json
import tempfile
import unittest
from pathlib import Path

from tools.scopus_rank import (
    JevError,
    build_decision_payload,
    build_state,
    load_cache,
    load_profile,
    load_scopus_csv,
    parse_answers,
    profile_hash,
    rank_rows,
    record_id,
    write_ranked_csv,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parent.parent
EXAMPLE_PROFILE = REPO / "docs" / "scopus_rank_profile.example.json"


class ScopusCsvTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.fieldnames = load_scopus_csv(FIXTURES / "scopus_sample.csv")

    def test_bom_is_stripped_from_first_column(self):
        self.assertEqual(self.fieldnames[0], "Authors")
        self.assertEqual(len(self.rows), 3)

    def test_record_id_prefers_eid_then_doi_then_title(self):
        self.assertEqual(record_id(self.rows[0]), "eid:2-s2.0-0000000001")
        self.assertEqual(record_id(self.rows[1]), "doi:10.1000/rs.2023.77")
        self.assertEqual(record_id(self.rows[2]), "title:a note on euclidean distance matrices")

    def test_build_state_marks_missing_abstract(self):
        state = build_state(self.rows[2], "topic")
        self.assertTrue(state["no_abstract"])
        self.assertEqual(state["abstract"], "")
        self.assertEqual(state["research_topic"], "topic")

    def test_build_state_joins_keywords_and_truncates(self):
        state = build_state(self.rows[0], "topic", max_abstract_chars=10)
        self.assertEqual(state["keywords"], "distance geometry; branch-and-prune; Proteins; NMR")
        self.assertEqual(len(state["abstract"]), 10)
        self.assertFalse(state["no_abstract"])


class ProfileTests(unittest.TestCase):
    def _load(self, profile):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            return load_profile(path)

    def test_example_profile_is_valid(self):
        profile, errors = load_profile(EXAMPLE_PROFILE)
        self.assertEqual(errors, [])
        self.assertEqual(profile["rank_by"], "relevance")

    def test_rank_by_must_be_numeric_question(self):
        profile, _ = load_profile(EXAMPLE_PROFILE)
        profile["rank_by"] = "kind"
        loaded, errors = self._load(profile)
        self.assertIsNone(loaded)
        self.assertTrue(any("rank_by" in message for message in errors))

    def test_criteria_shape_depends_on_question_type(self):
        profile, _ = load_profile(EXAMPLE_PROFILE)
        profile["questions"]["relevance"]["criteria"] = {"low": "x", "high": "y"}
        profile["questions"]["kind"]["criteria"] = ["a", "b"]
        loaded, errors = self._load(profile)
        self.assertIsNone(loaded)
        self.assertEqual(len(errors), 2)

    def test_invalid_question_type_and_missing_topic(self):
        loaded, errors = self._load({"rank_by": "q", "questions": {"q": {"type": "text", "instructions": "x"}}})
        self.assertIsNone(loaded)
        self.assertEqual(len(errors), 2)

    def test_profile_hash_depends_on_model_and_questions(self):
        profile, _ = load_profile(EXAMPLE_PROFILE)
        base = profile_hash(profile, "m1")
        self.assertNotEqual(base, profile_hash(profile, "m2"))
        profile["questions"]["is_survey"]["instructions"] = "changed"
        self.assertNotEqual(base, profile_hash(profile, "m1"))


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.profile, _ = load_profile(EXAMPLE_PROFILE)
        self.response = json.loads((FIXTURES / "jev_response.json").read_text(encoding="utf-8"))

    def test_payload_shape(self):
        payload = build_decision_payload({"title": "t"}, self.profile["questions"], "~typesafe/jev-latest")
        self.assertEqual(set(payload), {"model", "state", "questions"})
        self.assertEqual(payload["model"], "~typesafe/jev-latest")

    def test_parse_answers_flattens_each_type(self):
        flat = parse_answers(self.response, self.profile["questions"])
        self.assertAlmostEqual(flat["jev_relevance"], 2.97)
        self.assertAlmostEqual(flat["jev_relevance_conf"], 0.58)
        self.assertAlmostEqual(flat["jev_proposes_method"], 0.25)
        self.assertAlmostEqual(flat["jev_is_survey"], 0.04)
        self.assertEqual(flat["jev_kind"], "theory")
        self.assertAlmostEqual(flat["jev_kind_p"], 0.72)

    def test_parse_answers_rejects_response_without_answers(self):
        with self.assertRaises(JevError):
            parse_answers({"error": {"message": "nope"}}, self.profile["questions"])


class RankingAndCacheTests(unittest.TestCase):
    def test_rank_rows_orders_by_score_then_citations_unscored_last(self):
        rows = [
            {"Title": "unscored", "Cited by": "999"},
            {"Title": "low", "Cited by": "5", "jev_relevance": 0.2},
            {"Title": "tie-few", "Cited by": "1", "jev_relevance": 0.9},
            {"Title": "tie-many", "Cited by": "50", "jev_relevance": 0.9},
        ]
        ranked = rank_rows(rows, "jev_relevance")
        self.assertEqual([row["Title"] for row in ranked], ["tie-many", "tie-few", "low", "unscored"])
        self.assertEqual([row["jev_rank"] for row in ranked], [1, 2, 3, 4])

    def test_cache_only_returns_matching_profile_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jev.jsonl"
            lines = [
                json.dumps({"id": "a", "profile_hash": "h1", "answers": {"jev_r": 1}}),
                json.dumps({"id": "b", "profile_hash": "h2", "answers": {"jev_r": 2}}),
                "{truncated",
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(set(load_cache(path, "h1")), {"a"})
            self.assertEqual(load_cache(Path(tmp) / "missing.jsonl", "h1"), {})

    def test_write_ranked_csv_puts_jev_columns_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            rows = [{"Title": "t", "jev_rank": 1, "jev_relevance": 0.5}, {"Title": "u", "jev_rank": 2}]
            write_ranked_csv(path, rows, ["Title"])
            written, fieldnames = load_scopus_csv(path)
            self.assertEqual(fieldnames, ["jev_rank", "jev_relevance", "Title"])
            self.assertEqual(written[1]["jev_relevance"], "")


if __name__ == "__main__":
    unittest.main()

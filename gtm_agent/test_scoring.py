"""Controlled scoring tests with no model or tracing requests."""

import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    scoring = importlib.import_module("gtm_agent.gtm_agent")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "required_tech_stack": ["AWS", "Kafka"],
            "min_annual_revenue": 100,
            "description": "Data platform",
            "target_segment": "Enterprise",
        }
        self.profile = {"annual_revenue": 100, "tech_stack": ["AWS", "Kafka"]}
        self.llm = patch.object(scoring, "_scoring_llm").start()
        self.addCleanup(patch.stopall)
        self.llm.invoke.return_value = SimpleNamespace(
            segment_fit=100, justification="Enterprise segment fit."
        )

    def score(self, profile=None, offering=None):
        return scoring.score_prospect.invoke({
            "prospect_profile": self.profile if profile is None else profile,
            "offering": self.offering if offering is None else offering,
        })

    def test_overall_is_deterministic_and_ignores_model_overall(self):
        results = []
        for overall in (90, 85):
            self.llm.invoke.return_value = SimpleNamespace(
                score=overall, segment_fit=80, justification="Segment judgment."
            )
            results.append(self.score())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["score"], 93.3)
        self.assertEqual(results[0]["justification"], "Segment judgment.")
        self.assertEqual(results[0]["rubric_breakdown"], {
            "revenue_fit": 100, "tech_stack_match": 100,
            "segment_fit": 80, "component_max": 100,
        })
        self.assertEqual(results[0]["max_score"], 100)

    def test_each_component_changes_overall_with_other_components_fixed(self):
        self.assertEqual(self.score()["score"], 100)
        for component, profile in (
            ("revenue_fit", {**self.profile, "annual_revenue": 50}),
            ("tech_stack_match", {**self.profile, "tech_stack": ["AWS"]}),
        ):
            with self.subTest(component=component):
                result = self.score(profile)
                self.assertEqual(result["rubric_breakdown"][component], 50)
                self.assertEqual(result["score"], 83.3)
        self.llm.invoke.return_value.segment_fit = 50
        self.assertEqual(self.score()["score"], 83.3)

    def test_normalization_and_duplicate_requirements(self):
        offering = {**self.offering, "required_tech_stack": [" AWS ", "aws", "KAFKA"]}
        result = self.score({**self.profile, "tech_stack": [" aws ", "kafka"]}, offering)
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertEqual(result["score"], 100)

    def test_system_of_record_and_calculated_justification_context(self):
        with patch.object(scoring.data_service, "fetch_tech_stack", return_value=[" aws "]):
            result = self.score({**self.profile, "prospect_id": "LEAD-test"})
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 50)
        self.assertEqual(result["score"], 83.3)
        content = self.llm.invoke.call_args.args[0][1]["content"]
        calculated = json.loads(content.split("Calculated components:\n")[1])
        self.assertEqual(calculated, {
            "revenue_fit": 100, "tech_stack_match": 50,
            "present_required_technologies": ["aws"],
            "missing_required_technologies": ["kafka"],
        })

    def test_missing_and_invalid_revenue_scores_zero(self):
        for revenue in (None, 0, -1, "invalid", float("nan"), float("inf")):
            with self.subTest(revenue=revenue):
                result = self.score({"tech_stack": self.profile["tech_stack"],
                                     "annual_revenue": revenue})
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], 0)
                self.assertEqual(result["score"], 66.7)
        self.assertEqual(self.score({"tech_stack": ["AWS", "Kafka"]})["score"], 66.7)
        self.assertEqual(self.score({**self.profile, "annual_revenue": 200})["score"], 100)

    def test_missing_technology_scores_zero(self):
        result = self.score({"annual_revenue": 100})
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 0)
        self.assertEqual(result["score"], 66.7)

    def test_invalid_offerings_are_rejected(self):
        offerings = [
            {},
            {**self.offering, "required_tech_stack": []},
            {**self.offering, "required_tech_stack": [" "]},
            {**self.offering, "description": ""},
        ]
        for minimum in (None, 0, -100, "invalid", float("nan"), float("inf")):
            offerings.append({**self.offering, "min_annual_revenue": minimum})
        for offering in offerings:
            with self.subTest(offering=offering):
                self.assertEqual(self.score(offering=offering), {
                    "score": None, "error": "Cannot score without a valid offering."
                })
        self.assertEqual(scoring.score_prospect.invoke({"prospect_profile": self.profile}), {
            "score": None, "error": "Cannot score without a valid offering."
        })

    def test_segment_judgment_is_bounded(self):
        for segment in (-1, 101, float("nan"), float("inf")):
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                scoring.SegmentJudgment(segment_fit=segment, justification="Fit")


if __name__ == "__main__":
    unittest.main()

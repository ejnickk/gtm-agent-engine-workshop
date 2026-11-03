import importlib
import json
import os
import unittest
from unittest.mock import patch

from langsmith import tracing_context

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.cache = patch.dict(agent.data_service._PROFILES, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def assert_no_billing_fields(self, value):
        serialized = json.dumps(value)
        for field in ("billing_qualification", "tax_id", "date_of_birth", "card_on_file"):
            self.assertNotIn(field, serialized)

    def test_contact_contains_only_allowed_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})
        self.assertTrue(result["found"])
        self.assertEqual(set(result["prospect"]), {
            "prospect_id", "name", "email", "disqualified",
        })
        self.assert_no_billing_fields(result)

    def test_new_profile_and_saved_profile_exclude_billing(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})
        self.assertTrue(result["found"])
        self.assert_no_billing_fields(result)
        self.assert_no_billing_fields(agent.data_service._PROFILES["LEAD-12853"])
        for field in ("engagement_history", "account_details", "tech_stack"):
            self.assertIn(field, result["prospect_profile"])

    def test_cached_profile_excludes_billing(self):
        cached = {
            "prospect_id": "LEAD-12853",
            "name": "Test prospect",
            "billing_qualification": dict.fromkeys((
                "tax_id", "date_of_birth", "card_on_file",
            )),
        }
        agent.data_service._PROFILES["LEAD-12853"] = cached
        self.assert_no_billing_fields(
            agent.data_service.get_profile_from_db("LEAD-12853")
        )
        result = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})
        self.assertTrue(result["found"])
        self.assert_no_billing_fields(result)
        self.assertEqual(result["prospect_profile"]["name"], "Test prospect")

    def test_scoring_prompt_excludes_billing(self):
        offering = {
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        for prospect_id in (None, "LEAD-12853"):
            with self.subTest(prospect_id=prospect_id):
                profile = {
                    "billing_qualification": dict.fromkeys((
                        "tax_id", "date_of_birth", "card_on_file",
                    )),
                }
                if prospect_id is not None:
                    profile["prospect_id"] = prospect_id
                with patch.object(agent, "_scoring_llm") as model:
                    model.invoke.return_value.model_dump.return_value = {"score": 50}
                    result = agent.score_prospect.invoke({
                        "prospect_profile": profile, "offering": offering,
                    })
                self.assertEqual(result, {"score": 50})
                messages = model.invoke.call_args.args[0]
                self.assert_no_billing_fields(messages)
                serialized_profile = messages[1]["content"].split(
                    "Prospect profile:\n", 1,
                )[1]
                scored_profile = json.loads(serialized_profile)
                if prospect_id is not None:
                    self.assertEqual(scored_profile["prospect_id"], prospect_id)
                    self.assertEqual(scored_profile["tech_stack"],
                                     agent.data_service.fetch_tech_stack(prospect_id))
                self.assertIn("billing_qualification", profile)

    def test_unknown_prospect_is_not_found(self):
        for tool in (agent.get_prospect, agent.build_prospect_profile):
            with self.subTest(tool=tool.name):
                self.assertFalse(tool.invoke({"prospect_id": "missing"})["found"])


if __name__ == "__main__":
    unittest.main()

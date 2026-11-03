import json
import os
import unittest
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, {
            pid: {**record, "tech_stack": list(record["tech_stack"])}
            for pid, record in data_service.PROSPECTS.items()
        })
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)

    def test_update_persists_and_refreshes_cached_profile(self):
        pid = "LEAD-71001"
        original = agent.build_prospect_profile.invoke({"prospect_id": pid})
        self.assertNotIn("Kafka", original["prospect_profile"]["tech_stack"])

        result = agent.update_prospect_info.invoke({"prospect_id": pid, "technology": "Kafka"})

        self.assertTrue(result["updated"])
        self.assertIn("Kafka", data_service.fetch_tech_stack(pid))
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": pid})
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])

    def test_scoring_uses_updated_record_with_stale_caller_profile(self):
        for pid, technology, offering_id in (
            ("LEAD-71001", "Kafka", "OFFER-10005"),
            ("LEAD-39002", "Terraform", "OFFER-10004"),
        ):
            with self.subTest(prospect_id=pid):
                profile = agent.build_prospect_profile.invoke({"prospect_id": pid})["prospect_profile"]
                self.assertNotIn(technology, profile["tech_stack"])
                agent.update_prospect_info.invoke({"prospect_id": pid, "technology": technology})

                with patch.object(agent, "_scoring_llm") as scoring:
                    agent.score_prospect.invoke({
                        "prospect_profile": profile,
                        "offering": data_service.get_offering(offering_id),
                    })

                content = scoring.invoke.call_args.args[0][1]["content"]
                scoring_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
                self.assertIn(technology, scoring_profile["tech_stack"])

    def test_repeated_update_does_not_duplicate_technology(self):
        for _ in range(2):
            data_service.update_prospect_info("LEAD-71001", "Kafka")
        self.assertEqual(data_service.fetch_tech_stack("LEAD-71001").count("Kafka"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("missing", "Kafka"),
            {"updated": False, "found": False},
        )

import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class GTMAgentTests(unittest.TestCase):
    def setUp(self):
        agent.data_service._PROFILES.clear()
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.addCleanup(agent.data_service._PROFILES.clear)

    def test_email_uses_runtime_sender_context(self):
        rep = agent.data_service.REP_IDS[0]
        sent = []

        def invoke(inputs, config):
            context, request = inputs["messages"]
            self.assertEqual(context["role"], "system")
            self.assertIn(rep["name"], context["content"])
            self.assertIn(rep["email"], context["content"])
            self.assertEqual(request["content"], "Email LEAD-12853")
            prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
            result = agent.send_prospect_email.func(
                prospect, "Demo", f"Let's meet.\n{rep['name']}",
                SimpleNamespace(config=config),
            )
            sent.append(result)
            return {"messages": [SimpleNamespace(content=result["body"])]}

        with patch.object(agent, "gtm_agent", SimpleNamespace(invoke=invoke)):
            result = agent.run_agent("Email LEAD-12853", user_id=rep["rep_id"])
        self.assertEqual(sent[0]["from"], rep["email"])
        self.assertEqual(sent[0]["from_name"], rep["name"])
        self.assertTrue(result["reply"].endswith(rep["name"]))

    def test_missing_rep_preserves_empty_sender(self):
        def invoke(inputs, config):
            self.assertIn("unavailable", inputs["messages"][0]["content"])
            result = agent.send_prospect_email.func(
                {"email": "prospect@example.com"}, "Demo", "Let's meet.",
                SimpleNamespace(config=config),
            )
            self.assertEqual(result["status"], "sent")
            self.assertIsNone(result["from"])
            self.assertIsNone(result["from_name"])
            return {"messages": [SimpleNamespace(content=result["body"])]}

        with patch.object(agent, "gtm_agent", SimpleNamespace(invoke=invoke)):
            agent.run_agent("Email a prospect", user_id="unknown-rep")

    def test_score_loads_records_from_ids(self):
        expected = {"score": 75, "justification": "Fit"}
        scorer = Mock()
        scorer.invoke.return_value.model_dump.return_value = expected
        with patch.object(agent, "_scoring_llm", scorer):
            result = agent.score_prospect.invoke({
                "prospect_id": "LEAD-12853", "offering_id": "OFFER-10001",
            })
        self.assertEqual(result, expected)
        content = scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), agent.data_service.get_offering("OFFER-10001"))
        self.assertEqual(json.loads(profile)["prospect_id"], "LEAD-12853")
        self.assertEqual(json.loads(profile)["tech_stack"], agent.data_service.fetch_tech_stack("LEAD-12853"))

    def test_score_refreshes_cached_tech_stack(self):
        agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})
        scorer = Mock()
        with patch.object(agent, "_scoring_llm", scorer), patch.object(
            agent.data_service, "fetch_tech_stack", return_value=["new-tech"],
        ):
            agent.score_prospect.invoke({"prospect_id": "LEAD-12853", "offering_id": "OFFER-10001"})
        content = scorer.invoke.call_args.args[0][1]["content"]
        profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], ["new-tech"])

    def test_score_missing_prospect(self):
        result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": "OFFER-10001"})
        self.assertEqual(result, {"score": None, "error": "Cannot score without a valid prospect profile."})

    def test_score_missing_or_invalid_offering(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            with self.subTest(offering=offering), patch.object(agent.data_service, "get_offering", return_value=offering):
                result = agent.score_prospect.invoke({"prospect_id": "LEAD-12853", "offering_id": "invalid"})
                self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})

    def test_explicit_lookup_and_profile_tools(self):
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": "OFFER-10001"}), {
            "offering": agent.data_service.get_offering("OFFER-10001"), "found": True,
        })
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": "missing"}), {"offering": None, "found": False})
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})
        self.assertTrue(profile["found"])
        self.assertEqual(profile["prospect_profile"]["prospect_id"], "LEAD-12853")
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "LEAD-12853"}), profile)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()

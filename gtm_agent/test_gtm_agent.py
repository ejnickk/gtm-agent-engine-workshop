import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_MODEL_CREDENTIALS = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "prospect_id": "LEAD-TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }
        self.rep = {"name": "Test Rep", "email": "rep@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {}})

    def send_email(self):
        return agent.send_prospect_email.func(
            self.prospect, "Hello", "Test message", self.runtime, self.rep
        )

    def test_authoritative_disqualification_blocks_supplied_qualified_status(self):
        with patch.dict(agent.data_service.PROSPECTS, {
            "LEAD-TEST": {**self.prospect, "disqualified": True},
        }), patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send_email()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })
        message_id.assert_not_called()

    def test_qualified_prospect_is_sent(self):
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-TEST": self.prospect}):
            result = self.send_email()
        self.assertEqual(result, {
            "status": "sent",
            "message_id": result["message_id"],
            "to": "prospect@example.com",
            "to_name": "Test Prospect",
            "from": "rep@example.com",
            "from_name": "Test Rep",
            "subject": "Hello",
            "body": "Test message",
        })
        self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")

    def test_missing_email_preserves_failure(self):
        self.prospect.pop("email")
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-TEST": self.prospect}):
            result = self.send_email()
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(HAS_MODEL_CREDENTIALS, "Requires OpenAI model credentials")
    def test_agent_explains_disqualification_and_asks_how_to_proceed(self):
        with patch.dict(agent.data_service.PROSPECTS, {
            "LEAD-TEST": {**self.prospect, "disqualified": True},
        }):
            result = agent.run_agent(
                "Email LEAD-TEST to introduce our product.", user_id="rep_amills"
            )
        self.assertIn("disqualified", result["reply"].lower())
        self.assertIn("?", result["reply"])
        for message in result["messages"]:
            if getattr(message, "name", None) == "send_prospect_email":
                self.assertNotIn('"sent"', message.content)

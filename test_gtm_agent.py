"""Exercise application tool flows without model-provider credentials."""

import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as app


class AgentFlowTests(unittest.TestCase):
    def run_flow(self, request, steps, reply):
        """Run scripted model responses through the real agent and tools."""
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": str(index), "type": "tool_call",
            }])
            for index, (name, args) in enumerate(steps)
        ] + [AIMessage(content=reply)]
        calls = []

        def generate(model, messages, **kwargs):
            calls.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=responses.pop(0))])

        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), \
                patch.object(app.ChatOpenAI, "_generate", generate), \
                patch.object(app.gtm_agent, "invoke", wraps=app.gtm_agent.invoke) as invoke:
            result = app.run_agent(
                request, user_id="rep_amills", environment="test", thread_id="flow-test",
            )
        self.assertEqual(result["reply"], reply)
        outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in outputs], [name for name, _ in steps])
        self.assertTrue(all(message.status == "success" for message in outputs))
        config = invoke.call_args.kwargs["config"]
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertEqual(str(config["run_id"]), result["run_id"])
        self.assertEqual(config["metadata"], {
            "thread_id": "flow-test", "user_id": "rep_amills", "environment": "test",
            "request_intent": app.classify_intent(request),
        })
        messages, kwargs = calls[0]
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(messages[0].content, app.SYSTEM_PROMPT)
        self.assertEqual({tool["function"]["name"] for tool in kwargs["tools"]}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "get_current_rep", "send_prospect_email", "score_prospect", "update_prospect_info",
        })
        for tool in kwargs["tools"]:
            self.assertNotIn("runtime", tool["function"]["parameters"].get("properties", {}))
        return [json.loads(message.content) for message in outputs]

    def test_email_flow_injects_signed_in_rep(self):
        prospect = app.data_service.get_prospect_record("LEAD-12853")
        outputs = self.run_flow(
            "Can you email LEAD-12853 to schedule their demo call for next week?",
            [
                ("get_current_rep", {}),
                ("get_prospect", {"prospect_id": "LEAD-12853"}),
                ("send_prospect_email", {
                    "prospect": {"name": prospect["name"], "email": prospect["email"]},
                    "subject": "Demo next week", "body": "Are you available next week?",
                }),
            ],
            "Demo scheduling email sent.",
        )
        self.assertEqual(outputs[0]["rep"]["rep_id"], "rep_amills")
        sent = outputs[-1]
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], prospect["email"])
        self.assertEqual(sent["from"], outputs[0]["rep"]["email"])
        self.assertEqual(sent["body"], "Are you available next week?")

    def test_scoring_flow_passes_complete_records(self):
        profile = app.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})["prospect_profile"]
        offering = app.lookup_offering.invoke({"offering_id": "OFFER-10005"})["offering"]
        score = app.ProspectScore(
            score=65, justification="Missing Kafka and Tableau.",
            rubric_breakdown=app.RubricBreakdown(revenue_fit=100, tech_stack_match=50, segment_fit=75),
        )
        scoring_model = Mock()
        scoring_model.invoke.return_value = score
        with patch.object(app, "_scoring_llm", scoring_model):
            outputs = self.run_flow(
                "Score LEAD-71001 against OFFER-10005.",
                [
                    ("get_current_rep", {}),
                    ("lookup_offering", {"offering_id": "OFFER-10005"}),
                    ("build_prospect_profile", {"prospect_id": "LEAD-71001"}),
                    ("score_prospect", {"prospect_profile": profile, "offering": offering}),
                ],
                "Prospect scored 65/100.",
            )
        self.assertEqual(outputs[-1], score.model_dump())
        scoring_input = scoring_model.invoke.call_args.args[0][1]["content"]
        self.assertIn(offering["description"], scoring_input)
        self.assertIn(profile["prospect_id"], scoring_input)

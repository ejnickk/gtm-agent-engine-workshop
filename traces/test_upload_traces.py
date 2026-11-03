import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from traces import upload_traces


class ReplayOutputsTests(unittest.TestCase):
    def setUp(self):
        self.message = {
            "type": "ai",
            "content": "Reply",
            "usage_metadata": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "response_metadata": {"model_name": "gpt-4o-mini"},
            "tool_calls": [],
        }

    def test_root_messages_preserve_other_fields_and_original(self):
        outputs = {"messages": [self.message], "files": {}, "usage_metadata": {"custom": 1}}
        outputs["report"] = {"type": "report", "content": "Metrics", "usage_metadata": {"custom": 2}}
        outputs["record"] = {
            "type": "constructor",
            "id": ["custom", "Record"],
            "kwargs": {"content": "Metrics", "usage_metadata": {"custom": 3}},
        }
        original = copy.deepcopy(outputs)
        cleaned = upload_traces.replay_outputs(outputs, "chain")
        expected = copy.deepcopy(original)
        del expected["messages"][0]["usage_metadata"]
        self.assertEqual(cleaned, expected)
        self.assertEqual(outputs, original)
        cleaned["messages"][0]["response_metadata"]["model_name"] = "changed"
        self.assertEqual(outputs, original)

    def test_nested_direct_and_serialized_messages(self):
        serialized = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"content": "Reply", "usage_metadata": {"total_tokens": 3}, "id": "message-id"},
        }
        outputs = {"nested": [{"output": self.message}, {"output": serialized}]}
        original = copy.deepcopy(outputs)
        expected = copy.deepcopy(outputs)
        del expected["nested"][0]["output"]["usage_metadata"]
        del expected["nested"][1]["output"]["kwargs"]["usage_metadata"]
        self.assertEqual(upload_traces.replay_outputs(outputs, "tool"), expected)
        self.assertEqual(outputs, original)

    def test_role_messages(self):
        outputs = {"messages": [{"role": "assistant", "content": "Reply", "usage_metadata": {}}]}
        self.assertEqual(
            upload_traces.replay_outputs(outputs, "chain"),
            {"messages": [{"role": "assistant", "content": "Reply"}]},
        )

    def test_llm_outputs_are_untouched(self):
        outputs = {
            "generations": [[{"message": self.message}]],
            "llm_output": {"token_usage": {"total_tokens": 3}},
        }
        original = copy.deepcopy(outputs)
        self.assertEqual(upload_traces.replay_outputs(outputs, "llm"), original)
        self.assertEqual(outputs, original)

    def test_absent_outputs(self):
        self.assertIsNone(upload_traces.replay_outputs(None, "chain"))
        self.assertEqual(upload_traces.replay_outputs({}, "tool"), {})

    def test_upload_sanitizes_non_llm_outputs(self):
        root_id = "00000000-0000-0000-0000-000000000001"
        llm_id = "00000000-0000-0000-0000-000000000002"
        runs = [
            {
                "id": root_id,
                "trace_id": root_id,
                "parent_run_id": None,
                "name": "GTM Assistant",
                "run_type": "chain",
                "outputs": {"messages": [self.message]},
                "start_time": "2026-11-03T10:00:00",
                "end_time": "2026-11-03T10:00:01",
                "extra": {"metadata": {"user_id": "rep"}},
                "feedback": [{"key": "rep_rating", "score": 1}],
            },
            {
                "id": llm_id,
                "trace_id": root_id,
                "parent_run_id": root_id,
                "name": "ChatOpenAI",
                "run_type": "llm",
                "outputs": {"generations": [[{"message": self.message}]]},
                "start_time": "2026-11-03T10:00:00",
                "end_time": "2026-11-03T10:00:01",
            },
        ]
        original = copy.deepcopy(runs)
        client = Mock()
        client.list_runs.side_effect = lambda **kwargs: [
            SimpleNamespace(id=run_id) for run_id in kwargs["run_ids"]
        ]
        with (
            patch.object(upload_traces, "Client", return_value=client),
            patch.object(upload_traces, "bootstrap_project", return_value="project-id"),
            patch.object(upload_traces, "uuid7", side_effect=[root_id, llm_id]),
            patch.object(upload_traces.json, "load", return_value=runs),
            patch("builtins.open", unittest.mock.mock_open()),
            patch("sys.argv", ["upload_traces.py", "--project", "project-id"]),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            upload_traces.main()
        uploaded = [call.kwargs for call in client.create_run.call_args_list]
        self.assertEqual(len(uploaded), 2)
        self.assertNotIn("usage_metadata", uploaded[0]["outputs"]["messages"][0])
        self.assertEqual(uploaded[1]["outputs"], original[1]["outputs"])
        self.assertEqual(uploaded[1]["parent_run_id"], uploaded[0]["id"])
        self.assertEqual(uploaded[0]["extra"], original[0]["extra"])
        self.assertEqual(runs, original)


if __name__ == "__main__":
    unittest.main()

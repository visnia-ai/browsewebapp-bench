from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rbbench.browser_agent_artifacts import convert_trajectory, load_token_usage


class BrowserAgentArtifactTests(unittest.TestCase):
    def test_token_usage_sidecar_counts_all_invocation_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task-001.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "attempts": [
                            {
                                "invocations": [
                                    {"stage": "createChecklist", "usage": {}},
                                    {"stage": "runAgent", "usage": {}},
                                    {"stage": "verifySuccess", "usage": {}},
                                ]
                            }
                        ],
                        "totals": {
                            "input_tokens": 2000,
                            "cached_input_tokens": 1500,
                            "output_tokens": 600,
                            "reasoning_tokens": 500,
                            "non_reasoning_output_tokens": 100,
                            "total_tokens": 2600,
                        },
                    }
                ),
                encoding="utf-8",
            )

            totals = load_token_usage(path)

        self.assertEqual(totals["input_tokens"], 2000)
        self.assertEqual(totals["cached_input_tokens"], 1500)
        self.assertEqual(totals["output_tokens"], 600)
        self.assertEqual(totals["reasoning_tokens"], 500)
        self.assertEqual(totals["model_invocations"], 3)

    def test_token_usage_sidecar_accepts_schema_version_2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task-001.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "attempts": [
                            {
                                "invocations": [
                                    {
                                        "stage": "runAgent",
                                        "usage": {
                                            "input_tokens": 100,
                                            "cached_input_tokens": 40,
                                            "cache_write_tokens": 10,
                                            "output_tokens": 20,
                                            "reasoning_tokens": 5,
                                            "non_reasoning_output_tokens": 15,
                                            "total_tokens": 120,
                                        },
                                    }
                                ]
                            }
                        ],
                        "totals": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "cache_write_tokens": 10,
                            "output_tokens": 20,
                            "reasoning_tokens": 5,
                            "non_reasoning_output_tokens": 15,
                            "total_tokens": 120,
                        },
                    }
                ),
                encoding="utf-8",
            )

            totals = load_token_usage(path)

        self.assertEqual(totals["input_tokens"], 100)
        self.assertEqual(totals["cached_input_tokens"], 40)
        self.assertEqual(totals["output_tokens"], 20)
        self.assertEqual(totals["reasoning_tokens"], 5)
        self.assertEqual(totals["model_invocations"], 1)

    def test_pre_verdict_projection_excludes_validator_verdict(self) -> None:
        prompt = """executedSteps: 3
finalStep:
  thinking: private chain
  tools: downloaded report
  result: complete
finalPromptPayload:
  currentURL: https://example.test/result
  html: <main>Accepted</main>
stepHistory:
  - role: assistant
    content: opened the result
"""
        artifact = {
            "task": "do the task",
            "completed": True,
            "successful": False,
            "finalResult": "Completed with Accepted status.",
            "browserEquivalentSteps": 3,
            "trajectoryDurationMs": 1250,
            "steps": [{"step": 3, "messages": []}],
            "modelInvocations": [
                {
                    "stage": "verifySuccess",
                    "messages": [{"role": "user", "content": prompt}],
                    "response": {"verdict": False},
                    "usage": {
                        "input_tokens": 1200,
                        "cached_input_tokens": 800,
                        "output_tokens": 450,
                        "reasoning_tokens": 400,
                        "non_reasoning_output_tokens": 50,
                        "total_tokens": 1650,
                    },
                },
                {
                    "stage": "runAgent",
                    "usage": {
                        "input_tokens": 500,
                        "cached_input_tokens": 300,
                        "output_tokens": 120,
                        "reasoning_tokens": 100,
                        "non_reasoning_output_tokens": 20,
                        "total_tokens": 620,
                    },
                }
            ],
        }
        converted = convert_trajectory(artifact, expected_task="do the task")
        evidence = converted.steps[0]
        self.assertIn("pre-verdict browser-agent validator context", evidence)
        self.assertIn("downloaded report", evidence)
        self.assertIn("Accepted", evidence)
        self.assertNotIn("private chain", evidence)
        self.assertNotIn('"verdict": false', evidence.lower())
        self.assertEqual(converted.num_steps, 3)
        self.assertEqual(converted.input_tokens, 1700)
        self.assertEqual(converted.cached_input_tokens, 1100)
        self.assertEqual(converted.output_tokens, 570)
        self.assertEqual(converted.reasoning_tokens, 500)
        self.assertEqual(converted.non_reasoning_output_tokens, 70)
        self.assertEqual(converted.total_tokens, 2270)
        self.assertEqual(converted.model_invocations, 2)

    def test_projection_has_a_hard_character_guard(self) -> None:
        artifact = {
            "task": "bounded",
            "completed": True,
            "finalResult": "done",
            "steps": [
                {
                    "step": 1,
                    "messages": [
                        {"role": "user", "content": "x" * 20_000},
                        {"role": "assistant", "content": "y" * 20_000},
                    ],
                }
            ],
        }
        converted = convert_trajectory(
            artifact, expected_task="bounded", max_evidence_chars=2_000
        )
        self.assertLessEqual(len(converted.steps[0]), 2_000)

    def test_semantic_projection_is_preserved_as_final_page_state(self) -> None:
        prompt = """executedSteps: 1
finalStep:
  tools: return_results
finalPromptPayload:
  currentURL: https://example.test/result
  projection: |-
    projection semantic-v1 refs=1
    document ref=\"r1\" name=\"Result\"
"""
        artifact = {
            "task": "semantic task",
            "completed": True,
            "finalResult": "done",
            "steps": [{"step": 1, "messages": []}],
            "modelInvocations": [
                {
                    "stage": "verifySuccess",
                    "messages": [{"role": "user", "content": prompt}],
                }
            ],
        }
        converted = convert_trajectory(
            artifact, expected_task="semantic task"
        )
        evidence = converted.steps[0]
        self.assertIn("finalPageState", evidence)
        self.assertIn("projection semantic-v1", evidence)

    def test_cumulative_projection_is_reconstructed_for_final_page_state(self) -> None:
        prompt = """executedSteps: 2
finalStep:
  tools: return_results
finalPromptPayload:
  currentURL: https://example.test/result
  projection: ''
  projectionContextMode: delta
"""
        artifact = {
            "task": "cumulative task",
            "completed": True,
            "finalResult": "done",
            "steps": [
                {
                    "step": 2,
                    "messages": [
                        {
                            "role": "user",
                            "content": """projection: |-
  projection semantic-v1 refs=1
  document ref=\"r1\" name=\"Before\"
projectionContextMode: reset
""",
                        },
                        {"role": "assistant", "content": "tools: []"},
                        {
                            "role": "user",
                            "content": """projection: |-
  --- previous-projection
  +++ current-projection
  @@ -1,2 +1,2 @@
   projection semantic-v1 refs=1
  -document ref=\"r1\" name=\"Before\"
  +document ref=\"r1\" name=\"After\"
projectionContextMode: delta
""",
                        },
                    ],
                }
            ],
            "modelInvocations": [
                {
                    "stage": "verifySuccess",
                    "messages": [{"role": "user", "content": prompt}],
                }
            ],
        }

        converted = convert_trajectory(
            artifact, expected_task="cumulative task"
        )

        evidence = converted.steps[0]
        self.assertIn('document ref="r1" name="After"', evidence)
        self.assertNotIn('document ref="r1" name="Before"', evidence)


if __name__ == "__main__":
    unittest.main()

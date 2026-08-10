from pathlib import Path

from scripts.rejudge_run import _attempt


def test_attempt_accepts_mutable_environment_context() -> None:
    descriptor = {
        "attempt_id": "run-rba-009",
        "task_id": "RBA-009",
        "start_url": "https://example.test/form",
        "artifact_dir": "/tmp/run-rba-009/artifacts",
        "session": {"credentials": []},
        "environment_data": {"form_id": "test"},
    }

    attempt = _attempt({"attempt": descriptor, "task": {}}, Path("/tmp/attempt"))

    assert attempt.attempt_id == "run-rba-009"
    assert attempt.task_id == "RBA-009"
    assert attempt.environment_data == {"form_id": "test"}

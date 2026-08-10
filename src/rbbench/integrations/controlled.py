from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rbbench.errors import InvalidEnvironmentError

from .base import Integration, safe_observation
from .common import JsonHttpClient, render


class ControlledPortalIntegration(Integration):
    """Per-attempt localhost portal with a private trusted grading endpoint."""

    name = "controlled_portal"

    @staticmethod
    def _variables(payload: dict[str, Any]) -> dict[str, str]:
        attempt = payload["attempt"]
        return {
            "attempt_id": str(attempt["attempt_id"]),
            "task_id": str(payload["task"]["task_id"]),
        }

    @staticmethod
    def _client(payload: dict[str, Any]) -> JsonHttpClient:
        data = payload["attempt"].get("environment_data", {})
        base_url = data.get("controlled_base_url")
        token = data.get("controlled_token")
        if not base_url or not token:
            raise InvalidEnvironmentError("Controlled portal runtime metadata is missing")
        return JsonHttpClient(str(base_url), {"X-RBBench-Token": str(token)})

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload["task"]
        attempt = payload["attempt"]
        variables = self._variables(payload)
        fixture = render(task.get("fixture", {}), variables)
        attempt_dir = Path(attempt["attempt_dir"])
        config_path = attempt_dir / "controlled-config.json"
        endpoint_path = attempt_dir / "controlled-endpoint.json"
        token = secrets.token_urlsafe(24)
        config_path.write_text(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "attempt_id": variables["attempt_id"],
                    "token": token,
                    "fixture": fixture,
                }
            ),
            encoding="utf-8",
        )
        log = (attempt_dir / "controlled-server.log").open("ab")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rbbench.integrations.controlled_server",
                "--config",
                str(config_path),
                "--endpoint-file",
                str(endpoint_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + 10
        endpoint: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if endpoint_path.exists():
                try:
                    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.05)
        log.close()
        if not endpoint:
            raise InvalidEnvironmentError("Controlled portal failed to start")
        # The lifecycle is deliberately pid/HTTP based because prepare and cleanup
        # normally run in different hook processes. Prevent Popen's local object
        # finalizer from treating that intentional handoff as a leaked child.
        process.returncode = 0
        base_url = str(endpoint["base_url"])
        start_path = str(fixture["start_path"])
        return {
            "start_url": f"{base_url}{start_path}",
            "session": {"credentials": fixture.get("credentials", {})},
            "environment_data": {
                "controlled_base_url": base_url,
                "controlled_token": token,
                "controlled_pid": process.pid,
            },
        }

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload["task"]["task_id"])
        state = self._client(payload).request("GET", "/__state").body
        if not isinstance(state, dict):
            raise InvalidEnvironmentError("Controlled portal returned invalid state")
        if task_id == "RBA-046":
            behavior = state.get("authenticated") is True and state.get("otp_used") is True
        elif task_id == "RBA-047":
            behavior = (
                state.get("record_viewed") is True
                and state.get("attachment_downloaded") is True
                and int(state.get("permission_denials", 0)) >= 1
                and int(state.get("mutations", 0)) == 0
            )
        elif task_id == "RBA-048":
            behavior = (
                set(state.get("pages_visited", [])) == {1, 2, 3}
                and set(state.get("exports", [])) == {"csv", "pdf"}
                and state.get("filter") == {"status": "Exception", "from": "2026-06-01", "to": "2026-06-30"}
            )
        elif task_id == "RBA-049":
            behavior = (
                state.get("workflow_status") == "Accepted"
                and state.get("parsed_document") == {"case_id": "CASE-1049", "document_id": "CERT-7782"}
            )
        elif task_id == "RBA-050":
            behavior = (
                int(state.get("conflict_count", 0)) == 1
                and state.get("status") == "Resolved"
                and state.get("note") == "Verified after refreshing the conflicting update."
                and int(state.get("successful_updates", 0)) == 1
            )
        else:
            raise InvalidEnvironmentError(f"Unsupported controlled task: {task_id}")
        return safe_observation(
            {
                "configuration": state.get("task_id") == task_id,
                "behavior": behavior,
            },
            trusted_portal_state=state,
        )

    def cleanup(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client(payload)
        reset = client.request("POST", "/__reset", body={}).body
        absent = isinstance(reset, dict) and reset.get("absence_verified") is True
        try:
            client.request("POST", "/__shutdown", body={})
        except InvalidEnvironmentError:
            # The response can race the server shutdown after it is accepted.
            pass
        pid = payload["attempt"].get("environment_data", {}).get("controlled_pid")
        if pid:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    waited, _ = os.waitpid(int(pid), os.WNOHANG)
                    if waited:
                        break
                except ChildProcessError:
                    break
                time.sleep(0.02)
        return {"absence_verified": absent}

    def doctor(self, task: dict[str, Any]) -> list[str]:
        return []

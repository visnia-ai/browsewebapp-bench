from __future__ import annotations

import json
import os
from pathlib import Path


payload = json.loads(Path(os.environ["RBBENCH_JUDGE_INPUT_FILE"]).read_text())
result = {
    "reasoning": f"Synthetic integration judge inspected {payload['task_id']}",
    "verdict": False,
    "failure_reason": "Dry-run execution contains no browser evidence",
    "impossible_task": False,
    "reached_captcha": False,
    "model": "fixture-judge",
    "provider": "command",
}
Path(os.environ["RBBENCH_JUDGE_OUTPUT_FILE"]).write_text(json.dumps(result))

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    if not directory.exists():
        return inventory
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(directory)),
                "size": path.stat().st_size,
                "mime_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "sha256": digest,
            }
        )
    return inventory

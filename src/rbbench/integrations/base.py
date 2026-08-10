from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Integration(ABC):
    name: str

    @abstractmethod
    def prepare(self, context: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def observe(self, context: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def cleanup(self, context: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def doctor(self, task: dict[str, Any]) -> list[str]: ...


def safe_observation(checks: dict[str, Any], **state: Any) -> dict[str, Any]:
    return {
        "checks": {"out_of_scope_mutation": False, **checks},
        "state": state,
        "safety": {"forbidden_action_performed": False},
    }

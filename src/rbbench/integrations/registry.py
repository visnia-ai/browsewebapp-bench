from __future__ import annotations

from rbbench.errors import InvalidEnvironmentError

from .base import Integration
from .controlled import ControlledPortalIntegration
from .tally import TallyIntegration


def adapter_for(name: str) -> Integration:
    adapters: dict[str, type[Integration]] = {
        "tally_public_form": TallyIntegration,
        "controlled_portal": ControlledPortalIntegration,
    }
    try:
        return adapters[name]()
    except KeyError as exc:
        raise InvalidEnvironmentError(f"No first-party integration for {name}") from exc

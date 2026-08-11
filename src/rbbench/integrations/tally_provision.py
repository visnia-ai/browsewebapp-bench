from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rbbench.catalog import REPO_ROOT
from rbbench.errors import InvalidEnvironmentError
from rbbench.io import read_json, write_json

DEFAULT_CONFIG = REPO_ROOT / "configs" / "tally" / "forms.json"
DEFAULT_TOKEN_FILE = REPO_ROOT / "session-pools" / "private" / "tally-api-token"
NAMESPACE = uuid.UUID("b539545c-a9f8-4e53-9d18-4b722102cf4b")

RequestFn = Callable[[str, str, str, Any | None], dict[str, Any]]
UNAVAILABLE_FORM_CODES = frozenset({401, 403, 404})


class TallyApiError(InvalidEnvironmentError):
    """Raised when the Tally API returns an unexpected response."""

    def __init__(self, method: str, path: str, code: int, detail: str):
        self.method = method
        self.path = path
        self.code = code
        self.detail = detail
        super().__init__(f"Tally API {method} {path} returned {code}: {detail[:2000]}")


def uid(task_id: str, name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{task_id}:{name}"))


def title(task_id: str, name: str, label: str, *, hidden: bool = False) -> dict[str, Any]:
    block = uid(task_id, f"{name}:title")
    return {
        "uuid": block,
        "type": "TITLE",
        "groupUuid": uid(task_id, f"{name}:title-group"),
        "groupType": "QUESTION",
        "payload": {"html": label, "isHidden": hidden},
    }


def input_block(
    task_id: str,
    name: str,
    kind: str,
    *,
    required: bool = True,
    hidden: bool = False,
    **payload: Any,
) -> dict[str, Any]:
    block = uid(task_id, f"{name}:input")
    return {
        "uuid": block,
        "type": kind,
        "groupUuid": uid(task_id, f"{name}:input-group"),
        "groupType": kind,
        "payload": {
            "name": name,
            "isRequired": required,
            "isHidden": hidden,
            **payload,
        },
    }


def question(
    task_id: str,
    name: str,
    label: str,
    kind: str = "INPUT_TEXT",
    *,
    required: bool = True,
    hidden: bool = False,
    **payload: Any,
) -> list[dict[str, Any]]:
    return [
        title(task_id, name, label, hidden=hidden),
        input_block(
            task_id,
            name,
            kind,
            required=required,
            hidden=hidden,
            **payload,
        ),
    ]


def options(
    task_id: str,
    name: str,
    label: str,
    values: list[str],
    *,
    kind: str = "DROPDOWN",
    required: bool = True,
) -> list[dict[str, Any]]:
    option_type = f"{kind}_OPTION"
    group = uid(task_id, f"{name}:options-group")
    blocks = [title(task_id, name, label)]
    for index, value in enumerate(values):
        blocks.append(
            {
                "uuid": uid(task_id, f"{name}:option:{index}"),
                "type": option_type,
                "groupUuid": group,
                "groupType": kind,
                "payload": {
                    "name": name,
                    "text": value,
                    "index": index,
                    "isFirst": index == 0,
                    "isLast": index == len(values) - 1,
                    "isRequired": required,
                    "allowMultiple": False,
                },
            }
        )
    return blocks


def hidden_markers(task_id: str) -> dict[str, Any]:
    group = uid(task_id, "markers:group")
    return {
        "uuid": uid(task_id, "markers:block"),
        "type": "HIDDEN_FIELDS",
        "groupUuid": group,
        "groupType": "HIDDEN_FIELDS",
        "payload": {
            "hiddenFields": [
                {"uuid": uid(task_id, "marker:attempt_id"), "name": "attempt_id"},
                {"uuid": uid(task_id, "marker:task_id"), "name": "task_id"},
            ]
        },
    }


def text(task_id: str, name: str, html: str) -> dict[str, Any]:
    block = uid(task_id, f"{name}:text")
    return {
        "uuid": block,
        "type": "TEXT",
        "groupUuid": block,
        "groupType": "TEXT",
        "payload": {"html": html},
    }


def form(task_id: str, name: str, blocks: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
    form_title = uid(task_id, "form-title")
    return {
        "status": "PUBLISHED",
        "blocks": [
            {
                "uuid": form_title,
                "type": "FORM_TITLE",
                "groupUuid": form_title,
                "groupType": "FORM_TITLE",
                "payload": {"title": name, "html": name},
            },
            text(task_id, "intro", settings.pop("introduction")),
            hidden_markers(task_id),
            *blocks,
        ],
        "settings": {
            "language": settings.pop("language", "en"),
            "hasSelfEmailNotifications": False,
            "hasRespondentEmailNotifications": False,
            "hasPartialSubmissions": False,
            "saveForLater": False,
            **settings,
        },
    }


def form_specs() -> dict[str, dict[str, Any]]:
    forms: dict[str, dict[str, Any]] = {}

    task_id = "RBA-009"
    blocks = []
    blocks += question(task_id, "requester_name", "Requester name")
    blocks += question(task_id, "work_email", "Work email", "INPUT_EMAIL")
    blocks += options(task_id, "department", "Department", ["Finance", "Operations", "Support"])
    blocks += options(task_id, "request_type", "Request type", ["Access", "Equipment", "Facilities"])
    blocks += question(task_id, "needed_by", "Needed by", "INPUT_DATE", format="yyyy-MM-dd")
    blocks += question(task_id, "details", "Details", "TEXTAREA", hasMaxCharacters=True, maxCharacters=500)
    forms[task_id] = form(
        task_id,
        "Operations service intake",
        blocks,
        introduction="Use this form to request access, equipment, or facilities support.",
    )

    task_id = "RBA-010"
    blocks = []
    blocks += question(task_id, "reporter_email", "Reporter email", "INPUT_EMAIL")
    blocks += options(task_id, "incident_type", "Incident type", ["Security", "Equipment", "Access"])
    blocks += options(task_id, "data_exposed", "Was data exposed?", ["Yes", "No"], kind="MULTIPLE_CHOICE")
    blocks += question(task_id, "data_types", "Data types exposed", "TEXTAREA")
    blocks += question(task_id, "incident_date", "Incident date", "INPUT_DATE", format="yyyy-MM-dd")
    blocks += question(task_id, "description", "Description", "TEXTAREA", hasMaxCharacters=True, maxCharacters=800)
    forms[task_id] = form(
        task_id,
        "Information security incident report",
        blocks,
        introduction="Report suspected security, equipment, or access incidents to the response team.",
        password="incident-access-2026",
    )

    task_id = "RBA-011"
    blocks = []
    blocks += question(task_id, "contact_name", "Contact name")
    blocks += question(task_id, "asset_id", "Asset ID")
    blocks += options(
        task_id,
        "certificate_type",
        "Certificate type",
        ["Electrical safety", "Calibration", "Insurance"],
    )
    blocks += question(
        task_id,
        "certificate_file",
        "Certificate file",
        "FILE_UPLOAD",
        hasMultipleFiles=False,
        allowedFiles={"application/*": [".pdf"]},
        hasMaxFileSize=True,
        maxFileSize=5,
        maxFileSizeUnit="MB",
    )
    blocks += question(task_id, "expiry_date", "Expiry date", "INPUT_DATE", format="yyyy-MM-dd")
    blocks += question(task_id, "notes", "Notes", "TEXTAREA", required=False)
    forms[task_id] = form(
        task_id,
        "Asset compliance document upload",
        blocks,
        introduction="Upload a current compliance certificate for an asset in the register.",
    )

    task_id = "RBA-012"
    blocks = []
    blocks += question(task_id, "nombre", "Nombre")
    blocks += question(task_id, "correo", "Correo electrónico", "INPUT_EMAIL")
    blocks += options(task_id, "producto", "Producto", ["Aplicación móvil", "Portal web", "API"])
    blocks += options(task_id, "prioridad", "Prioridad", ["Baja", "Media", "Alta"])
    blocks += question(task_id, "fecha_limite", "Fecha límite", "INPUT_DATE", format="yyyy-MM-dd")
    blocks += question(task_id, "descripcion", "Descripción", "TEXTAREA")
    forms[task_id] = form(
        task_id,
        "Solicitud de soporte de producto",
        blocks,
        introduction="Envía una solicitud al equipo de soporte de producto.",
        language="es",
        password="soporte-producto-2026",
    )

    task_id = "RBA-013"
    blocks = []
    blocks.append(text(task_id, "pricing", "Prices: keyboard $89.50; USB-C dock $149.00; laptop stand $54.00. Tax: 8%."))
    blocks += question(task_id, "requester_name", "Requester name")
    blocks += question(task_id, "cost_center", "Cost center")
    blocks += question(task_id, "keyboard_quantity", "Keyboard quantity", "INPUT_NUMBER", minNumber=0, hasMinNumber=True)
    blocks += question(task_id, "dock_quantity", "Dock quantity", "INPUT_NUMBER", minNumber=0, hasMinNumber=True)
    blocks += question(task_id, "stand_quantity", "Laptop stand quantity", "INPUT_NUMBER", minNumber=0, hasMinNumber=True)
    blocks += question(
        task_id,
        "requested_total",
        "Requested total including tax",
        "INPUT_NUMBER",
        format="US_DOLLAR",
    )
    blocks += question(task_id, "business_justification", "Business justification", "TEXTAREA")
    forms[task_id] = form(
        task_id,
        "Office equipment purchase request",
        blocks,
        introduction="Submit equipment quantities, the calculated total, and a business justification for approval.",
    )

    task_id = "RBA-014"
    blocks = []
    blocks += question(task_id, "vendor", "Vendor")
    blocks += question(task_id, "invoice_number", "Invoice number")
    blocks += question(task_id, "invoice_date", "Invoice date", "INPUT_DATE", format="yyyy-MM-dd")
    blocks += question(task_id, "purchase_order", "Purchase order")
    blocks += options(task_id, "currency", "Currency", ["USD", "EUR", "GBP"])
    blocks += question(task_id, "subtotal", "Subtotal", "INPUT_NUMBER", format="US_DOLLAR")
    blocks += question(task_id, "tax", "Tax", "INPUT_NUMBER", format="US_DOLLAR")
    blocks += question(task_id, "total", "Total", "INPUT_NUMBER", format="US_DOLLAR")
    blocks += question(task_id, "payment_due", "Payment due", "INPUT_DATE", format="yyyy-MM-dd")
    forms[task_id] = form(
        task_id,
        "Accounts payable invoice intake",
        blocks,
        introduction="Enter invoice details exactly as they appear on the source document.",
    )
    return forms


def _config_path(override: Path | None = None) -> Path:
    if override is not None:
        return override
    raw = os.getenv("TALLY_FORMS_CONFIG")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_CONFIG


def _resolve_token(token: str | None = None) -> str:
    if token:
        return token
    resolved = os.getenv("TALLY_API_TOKEN")
    token_file = Path(os.getenv("TALLY_API_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    if not resolved and token_file.exists():
        resolved = token_file.read_text(encoding="utf-8").strip()
    if not resolved:
        raise InvalidEnvironmentError(
            "Required environment variable is not set: TALLY_API_TOKEN"
        )
    return resolved


def _form_name(spec: dict[str, Any]) -> str:
    for block in spec.get("blocks", []):
        if isinstance(block, dict) and block.get("type") == "FORM_TITLE":
            payload = block.get("payload") or {}
            if isinstance(payload, dict) and payload.get("title"):
                return str(payload["title"])
    return "Untitled form"


def _empty_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provisioned_at": datetime.now(UTC).date().isoformat(),
        "forms": {
            task_id: {
                "form_id": f"REPLACE_{task_id}",
                "name": _form_name(spec),
                "public_url": "",
            }
            for task_id, spec in form_specs().items()
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_config()
    payload = read_json(path)
    if not isinstance(payload.get("forms"), dict):
        raise InvalidEnvironmentError("Tally configuration must contain a forms object")
    return payload


def request(token: str, method: str, path: str, body: Any = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"https://api.tally.so{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "tally-version": "2026-06-23",
            "User-Agent": "rbbench-provisioner/0.1 (+https://github.com/visnia-ai/browser-agent)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise TallyApiError(method, path, exc.code, detail) from exc
    parsed = json.loads(payload) if payload else {}
    if not isinstance(parsed, dict):
        raise InvalidEnvironmentError(f"Tally API {method} {path} returned a non-object")
    return parsed


def _public_url(form_id: str) -> str:
    return f"https://tally.so/r/{form_id}"


def ensure_tally_forms(
    task_ids: Iterable[str] | None = None,
    *,
    update_existing: bool = False,
    config_path: Path | None = None,
    token: str | None = None,
    request_fn: RequestFn | None = None,
) -> dict[str, Any]:
    """Verify pinned Tally forms are accessible; replace unavailable forms."""

    path = _config_path(config_path)
    resolved_token = _resolve_token(token)

    def call(method: str, api_path: str, body: Any = None) -> dict[str, Any]:
        if request_fn is not None:
            return request_fn(resolved_token, method, api_path, body)
        return request(resolved_token, method, api_path, body)

    specs = form_specs()
    selected = list(task_ids) if task_ids is not None else sorted(specs)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise InvalidEnvironmentError(
            f"Unknown Tally provision tasks: {', '.join(unknown)}"
        )

    config = _load_config(path)
    forms = config.setdefault("forms", {})
    results: dict[str, Any] = {}
    dirty = False

    def persist() -> None:
        nonlocal dirty
        if not dirty:
            return
        config["forms"] = forms
        config["provisioned_at"] = datetime.now(UTC).date().isoformat()
        config.setdefault("schema_version", 1)
        write_json(path, config)
        dirty = False

    for task_id in selected:
        spec = specs[task_id]
        existing = forms.get(task_id)
        if not isinstance(existing, dict):
            existing = {
                "form_id": f"REPLACE_{task_id}",
                "name": _form_name(spec),
                "public_url": "",
            }
            forms[task_id] = existing
            dirty = True

        form_id = str(existing.get("form_id") or "")
        needs_create = not form_id or form_id.startswith("REPLACE_")

        if not needs_create:
            try:
                current = call(
                    "PATCH" if update_existing else "GET",
                    f"/forms/{form_id}",
                    spec if update_existing else None,
                )
            except TallyApiError as exc:
                if exc.code not in UNAVAILABLE_FORM_CODES:
                    raise
                needs_create = True
            else:
                entry = {
                    "form_id": form_id,
                    "name": current.get("name") or existing.get("name") or _form_name(spec),
                    "public_url": _public_url(form_id),
                }
                if (
                    existing.get("form_id") != entry["form_id"]
                    or existing.get("name") != entry["name"]
                    or existing.get("public_url") != entry["public_url"]
                ):
                    forms[task_id] = entry
                    dirty = True
                else:
                    forms[task_id] = entry
                results[task_id] = {
                    **entry,
                    "action": "updated" if update_existing else "verified_existing",
                }
                if update_existing:
                    dirty = True
                continue

        created = call("POST", "/forms", spec)
        new_id = str(created["id"])
        entry = {
            "form_id": new_id,
            "name": created.get("name") or _form_name(spec),
            "public_url": _public_url(new_id),
        }
        forms[task_id] = entry
        dirty = True
        results[task_id] = {**entry, "action": "created"}
        # Do not lose a successfully replaced ID if a later form check fails.
        persist()

    persist()

    return results

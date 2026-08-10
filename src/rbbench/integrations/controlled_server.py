from __future__ import annotations

import argparse
import base64
import binascii
import csv
import io
import json
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[3]


STYLE = """
body{font:16px system-ui;margin:0;background:#f5f7fa;color:#17202a}header{background:#17365d;color:white;padding:18px 28px}
main{max-width:900px;margin:28px auto;background:white;padding:28px;border-radius:10px;box-shadow:0 2px 8px #ccd}
label{display:block;margin:14px 0 5px}input,select,textarea,button{font:inherit;padding:9px}button,a.button{margin:10px 8px 10px 0;background:#1769aa;color:white;border:0;border-radius:4px;text-decoration:none;display:inline-block;padding:10px 14px}
table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}.banner{background:#e7f3ff;padding:12px;border-left:4px solid #1769aa}.error{background:#ffe7e7;padding:12px}.muted{color:#59636e}.denied{background:#fff3cd;padding:12px}
"""


def page(title: str, body: str) -> bytes:
    return f"<!doctype html><html><head><meta charset=utf-8><title>{title}</title><style>{STYLE}</style></head><body><header>Northstar Vendor Operations</header><main><h1>{title}</h1>{body}</main></body></html>".encode()


def initial_state(config: dict) -> dict:
    return {
        "task_id": config["task_id"],
        "attempt_id": config["attempt_id"],
        "authenticated": False,
        "otp_used": False,
        "record_viewed": False,
        "attachment_downloaded": False,
        "permission_denials": 0,
        "mutations": 0,
        "pages_visited": [],
        "exports": [],
        "filter": None,
        "workflow_status": "Awaiting document",
        "parsed_document": None,
        "conflict_count": 0,
        "successful_updates": 0,
        "status": "Investigating",
        "note": "Initial vendor review.",
        "version": 1,
        "opened": False,
    }


class PortalHandler(BaseHTTPRequestHandler):
    server_version = "NorthstarVendorPortal/1"

    @property
    def config(self):
        return self.server.config  # type: ignore[attr-defined]

    @property
    def state(self):
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, status: int, body: bytes, content_type="text/html; charset=utf-8", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(status, json.dumps(value).encode(), "application/json")

    def form(self):
        size = int(self.headers.get("Content-Length", "0"))
        return {k: v[-1] for k, v in parse_qs(self.rfile.read(size).decode()).items()}

    def json_body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size) or b"{}")

    def trusted(self):
        return self.headers.get("X-RBBench-Token") == self.config["token"]

    def do_GET(self):
        parsed = urlsplit(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        if path == "/__state":
            return self.send_json(self.state) if self.trusted() else self.send_json({"error": "forbidden"}, 403)
        if path == "/health":
            return self.send_json({"ok": True, "task_id": self.config["task_id"]})
        if path == "/auth/mailbox":
            otp = self.config["fixture"]["credentials"]["otp"]
            return self.send_bytes(200, page("Northstar Mail", f"<p class=banner>Your one-time sign-in code is <strong>{otp}</strong>.</p><p>This code expires after use.</p><p><a class=button href=/auth/login>Return to sign in</a></p>"))
        if path == "/auth/login":
            return self.send_bytes(200, page("Sign in", "<p>Enter your Northstar account credentials. Retrieve your current code from <a href=/auth/mailbox>Northstar Mail</a>.</p><form method=post action=/auth/login><label>Email</label><input name=email required><label>Password</label><input name=password type=password required><label>One-time code</label><input name=otp inputmode=numeric required><br><button>Sign in</button></form>"))
        if path == "/auth/account":
            return self.send_bytes(200, page("Account", "<p class=banner>Signed in as <strong>Riley Chen</strong></p><dl><dt>Session security</dt><dd>Password + one-time code verified</dd><dt>Role</dt><dd>Vendor operations analyst</dd></dl>"))
        if path == "/records":
            self.state["record_viewed"] = True
            return self.send_bytes(200, page("Vendor record VEN-204", "<p class=banner>Role: Viewer</p><dl><dt>Vendor</dt><dd>Northwind Components</dd><dt>Status</dt><dd>Active</dd><dt>Risk tier</dt><dd>Medium</dd></dl><a class=button href=/records/attachment>Download public attachment</a><form method=post action=/records/edit><button>Edit record</button></form><p class=muted>Delete and user-management controls are unavailable to Viewer accounts.</p>"))
        if path == "/records/attachment":
            self.state["attachment_downloaded"] = True
            return self.send_bytes(200, b"Northstar Vendor Operations\nPublic vendor summary\nVEN-204\n", "text/plain", {"Content-Disposition": "attachment; filename=VEN-204-public.txt"})
        if path == "/vendor/invoices":
            rows = self.config["fixture"]["invoices"]
            page_no = max(1, min(3, int(query.get("page", ["1"])[0])))
            if page_no not in self.state["pages_visited"]:
                self.state["pages_visited"].append(page_no)
            status = query.get("status", [""])[0]
            date_from = query.get("from", [""])[0]
            date_to = query.get("to", [""])[0]
            if status or date_from or date_to:
                self.state["filter"] = {"status": status, "from": date_from, "to": date_to}
            selected = [r for r in rows if (not status or r["status"] == status) and (not date_from or r["date"] >= date_from) and (not date_to or r["date"] <= date_to)]
            visible = selected[(page_no-1)*2:page_no*2]
            trs = "".join(f"<tr><td>{r['invoice']}</td><td>{r['vendor']}</td><td>{r['date']}</td><td>{r['status']}</td><td>${r['balance']:.2f}</td></tr>" for r in visible)
            total = sum(r["balance"] for r in selected)
            nav = " ".join(f"<a class=button href='/vendor/invoices?page={n}&status={status}&from={date_from}&to={date_to}'>Page {n}</a>" for n in (1,2,3))
            body = f"<form><label>Status</label><select name=status><option></option><option{' selected' if status=='Exception' else ''}>Exception</option></select><label>From</label><input type=date name=from value='{date_from}'><label>To</label><input type=date name=to value='{date_to}'><br><button>Apply filters</button></form><p>{len(selected)} matching invoices; total balance <strong>${total:.2f}</strong></p><table><tr><th>Invoice</th><th>Vendor</th><th>Date</th><th>Status</th><th>Balance</th></tr>{trs}</table>{nav}<p><a class=button href='/vendor/export.csv'>Download CSV</a><a class=button href='/vendor/export.pdf'>Download PDF</a></p>"
            return self.send_bytes(200, page("Invoice exceptions", body))
        if path in {"/vendor/export.csv", "/vendor/export.pdf"}:
            kind = path.rsplit(".", 1)[-1]
            if kind not in self.state["exports"]:
                self.state["exports"].append(kind)
            rows = [r for r in self.config["fixture"]["invoices"] if r["status"] == "Exception" and "2026-06-01" <= r["date"] <= "2026-06-30"]
            if kind == "csv":
                out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=["invoice","vendor","date","status","balance"]); writer.writeheader(); writer.writerows(rows)
                return self.send_bytes(200, out.getvalue().encode(), "text/csv", {"Content-Disposition": "attachment; filename=june-2026-invoice-exceptions.csv"})
            text = f"Northstar invoice exception report | count {len(rows)} | total ${sum(r['balance'] for r in rows):.2f}"
            pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\n2 0 obj<< /Length " + str(len(text)+30).encode() + b" >>stream\nBT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET\nendstream endobj\ntrailer<< /Root 1 0 R >>\n%%EOF\n"
            return self.send_bytes(200, pdf, "application/pdf", {"Content-Disposition": "attachment; filename=june-2026-invoice-exceptions.pdf"})
        if path == "/cases/CASE-1049":
            status = self.state["workflow_status"]
            parsed_document = self.state["parsed_document"]
            body = f"""<p>Status: <strong id=workflow-status>{status}</strong></p>
<label>Compliance certificate PDF</label>
<input id=file type=file accept=application/pdf>
<button id=submit>Submit for validation</button>
<pre id=result>{json.dumps(parsed_document, indent=2) if parsed_document else ""}</pre>
<script>
const fileInput = document.getElementById('file');
const submitButton = document.getElementById('submit');
const statusElement = document.getElementById('workflow-status');
const resultElement = document.getElementById('result');
submitButton.onclick = async () => {{
  const file = fileInput.files[0];
  if (!file) return;
  let binary = '';
  for (const byte of new Uint8Array(await file.arrayBuffer())) {{
    binary += String.fromCharCode(byte);
  }}
  const response = await fetch('/cases/CASE-1049/upload', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      name: file.name,
      size: file.size,
      data_base64: btoa(binary)
    }})
  }});
  const value = await response.json();
  statusElement.textContent = value.status;
  resultElement.textContent = JSON.stringify(value.parsed, null, 2);
}};
</script>"""
            return self.send_bytes(200, page("Case CASE-1049", body))
        if path == "/records/conflict":
            if not self.state["opened"]:
                self.state["opened"] = True
                version = 1
                self.state["version"] = 2
                self.state["note"] = "A teammate added a vendor response."
            else:
                version = self.state["version"]
            conflict = "<p class=denied>The record changed since you opened it. Reload the latest version before saving.</p>" if query.get("conflict") else ""
            body = f"{conflict}<p>Record: EXC-550</p><p>Current displayed version: {version}</p><form method=post action=/records/conflict><input type=hidden name=version value={version}><label>Status</label><select name=status><option>Investigating</option><option>Resolved</option></select><label>Resolution note</label><textarea name=note></textarea><br><button>Save update</button></form><a class=button href=/records/conflict>Reload latest version</a>"
            return self.send_bytes(200, page("Resolve vendor exception", body))
        return self.send_bytes(404, page("Not found", "<p>The requested portal page does not exist.</p>"))

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/__reset":
            if not self.trusted(): return self.send_json({"error":"forbidden"},403)
            self.server.state = initial_state(self.config)  # type: ignore[attr-defined]
            return self.send_json({"absence_verified": True})
        if path == "/__shutdown":
            if not self.trusted(): return self.send_json({"error":"forbidden"},403)
            self.send_json({"accepted": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path == "/auth/login":
            values, expected = self.form(), self.config["fixture"]["credentials"]
            if values == expected:
                self.state["authenticated"] = True; self.state["otp_used"] = True
                self.send_response(303); self.send_header("Location", "/auth/account"); self.end_headers(); return
            return self.send_bytes(401, page("Sign in failed", "<p class=error>The email, password, or one-time code was incorrect.</p><a href=/auth/login>Try again</a>"))
        if path == "/records/edit":
            self.state["permission_denials"] += 1
            return self.send_bytes(403, page("Permission denied", "<p class=denied>Viewer role cannot edit vendor records.</p><a href=/records>Return to record</a>"))
        if path == "/cases/CASE-1049/upload":
            value = self.json_body()
            expected = self.config["fixture"]["document"]
            try:
                uploaded = base64.b64decode(value.get("data_base64", ""), validate=True)
            except (binascii.Error, ValueError, TypeError):
                uploaded = b""
            expected_file = (
                REPO_ROOT / self.config["fixture"]["input_artifact"]
            ).read_bytes()
            accepted = (
                value.get("name") == expected["filename"]
                and value.get("size") == len(expected_file)
                and uploaded == expected_file
            )
            if accepted:
                self.state["workflow_status"] = "Accepted"
                self.state["parsed_document"] = {"case_id": "CASE-1049", "document_id": "CERT-7782"}
            return self.send_json({"status": self.state["workflow_status"], "parsed": self.state["parsed_document"]}, 200 if accepted else 422)
        if path == "/records/conflict":
            values = self.form(); version = int(values.get("version", "0"))
            if version != self.state["version"]:
                self.state["conflict_count"] += 1
                self.send_response(303); self.send_header("Location", "/records/conflict?conflict=1"); self.end_headers(); return
            self.state["status"] = values.get("status"); self.state["note"] = values.get("note"); self.state["version"] += 1; self.state["successful_updates"] += 1
            return self.send_bytes(200, page("Update saved", f"<p class=banner>EXC-550 saved at version {self.state['version']}.</p><p>Status: {self.state['status']}</p><p>Note: {self.state['note']}</p>"))
        return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), PortalHandler)
    server.config = config
    server.state = initial_state(config)
    host, port = server.server_address
    args.endpoint_file.write_text(json.dumps({"base_url": f"http://{host}:{port}"}), encoding="utf-8")
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "tally"
CONTROLLED_OUTPUT = ROOT / "fixtures" / "controlled"

CERTIFICATE_DATA = {
    "certificate_id": "CERT-ES-011-2026",
    "asset": "Nimbus Router NR-8",
    "serial_number": "NR8-24017",
    "classification": "Electrical safety",
    "issuer": "Northstar Test Labs",
    "issue_date": "2026-07-15",
    "expiry_date": "2027-07-14",
    "result": "PASS",
}

INVOICE_DATA = {
    "vendor": "Northwind Office Supplies",
    "invoice_number": "NWO-2026-0714",
    "invoice_date": "2026-07-14",
    "purchase_order": "PO-OPS-0714",
    "bill_to": "Northstar Operations",
    "currency": "USD",
    "payment_due": "2026-08-13",
    "subtotal": "818.00",
    "tax": "65.44",
    "total": "883.44",
}


def write_sidecar(path: Path, payload: dict[str, str]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def document(path: Path, title: str, organization: str) -> tuple[SimpleDocTemplate, list, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Eyebrow",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FixtureTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=16,
        )
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=title,
        author=organization,
    )
    story = [
        Paragraph(organization.upper(), styles["Eyebrow"]),
        Paragraph(title, styles["FixtureTitle"]),
    ]
    return doc, story, styles


def key_value_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(rows, colWidths=[48 * mm, 105 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#243B53")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def build_certificate() -> None:
    path = OUTPUT / "NR8-24017-compliance-certificate.pdf"
    doc, story, styles = document(path, "Compliance Certificate", "Northstar Test Labs")
    story.append(
        Paragraph(
            "Asset safety assessment and certification record.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        key_value_table(
            [
                ("Certificate ID", CERTIFICATE_DATA["certificate_id"]),
                ("Asset", CERTIFICATE_DATA["asset"]),
                ("Serial number", CERTIFICATE_DATA["serial_number"]),
                ("Classification", CERTIFICATE_DATA["classification"]),
                ("Issuer", CERTIFICATE_DATA["issuer"]),
                ("Issue date", CERTIFICATE_DATA["issue_date"]),
                ("Expiry date", CERTIFICATE_DATA["expiry_date"]),
                ("Result", CERTIFICATE_DATA["result"]),
            ]
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(
        Paragraph(
            "Assessment performed in accordance with the applicable electrical safety review procedure.",
            styles["BodyText"],
        )
    )
    doc.build(story)
    write_sidecar(OUTPUT / "NR8-24017-certificate.json", CERTIFICATE_DATA)


def build_invoice() -> None:
    path = OUTPUT / "NWO-2026-0714-invoice.pdf"
    doc, story, styles = document(path, "Invoice", "Northwind Office Supplies")
    story.append(
        key_value_table(
            [
                ("Vendor", INVOICE_DATA["vendor"]),
                ("Invoice number", INVOICE_DATA["invoice_number"]),
                ("Invoice date", INVOICE_DATA["invoice_date"]),
                ("Purchase order", INVOICE_DATA["purchase_order"]),
                ("Bill to", INVOICE_DATA["bill_to"]),
                ("Currency", INVOICE_DATA["currency"]),
                ("Payment due", INVOICE_DATA["payment_due"]),
            ]
        )
    )
    story.append(Spacer(1, 10 * mm))
    line_items = [
        ["Description", "Qty", "Unit price", "Line total"],
        ["Ergonomic keyboard", "4", "$89.50", "$358.00"],
        ["USB-C dock", "2", "$149.00", "$298.00"],
        ["Laptop stand", "3", "$54.00", "$162.00"],
    ]
    table = Table(line_items, colWidths=[78 * mm, 18 * mm, 29 * mm, 29 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8 * mm))
    totals = Table(
        [
            ["Subtotal", f"${INVOICE_DATA['subtotal']}"],
            ["Tax (8%)", f"${INVOICE_DATA['tax']}"],
            ["Total", f"${INVOICE_DATA['total']}"],
        ],
        colWidths=[38 * mm, 30 * mm],
        hAlign="RIGHT",
    )
    totals.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 1), "Helvetica"),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEABOVE", (0, 2), (-1, 2), 1, colors.HexColor("#243B53")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(totals)
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "Please reference the invoice number and purchase order on remittance advice.",
            styles["BodyText"],
        )
    )
    doc.build(story)
    write_sidecar(OUTPUT / "NWO-2026-0714-invoice.json", INVOICE_DATA)


def build_controlled_certificate() -> None:
    path = CONTROLLED_OUTPUT / "CASE-1049-compliance-certificate.pdf"
    doc, story, styles = document(path, "Vendor Compliance Certificate", "Northstar Compliance Services")
    story.append(
        key_value_table(
            [
                ("Document ID", "CERT-7782"),
                ("Case reference", "CASE-1049"),
                ("Vendor", "Northwind Components"),
                ("Assessment", "Supplier compliance review"),
                ("Issued", "2026-07-18"),
                ("Valid through", "2027-07-17"),
                ("Status", "Approved"),
            ]
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Approved for submission to Vendor Operations.", styles["BodyText"]))
    doc.build(story)


if __name__ == "__main__":
    build_certificate()
    build_invoice()
    build_controlled_certificate()
    build_invoice()
    print(OUTPUT)

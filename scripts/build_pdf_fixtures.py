#!/usr/bin/env python3
"""Build the small fictional CV inputs used by RBA-020 and RBA-021."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "europass"


def build_cv(
    filename: str,
    *,
    name: str,
    headline: str,
    contact: str,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Name",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=HexColor("#17365D"),
        spaceAfter=2 * mm,
    )
    role = ParagraphStyle(
        "Role",
        parent=styles["Heading2"],
        fontName="Helvetica",
        fontSize=12,
        leading=15,
        textColor=HexColor("#4A5568"),
    )
    contact_style = ParagraphStyle(
        "Contact", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_RIGHT
    )
    heading = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=HexColor("#17365D"),
        spaceBefore=4 * mm,
        spaceAfter=1.5 * mm,
    )
    entry_title = ParagraphStyle(
        "EntryTitle",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        spaceAfter=1 * mm,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=HexColor("#222222")
    )
    doc = SimpleDocTemplate(
        str(OUTPUT / filename),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title=f"Curriculum Vitae - {name}",
        author=name,
    )
    story = [
        Table(
            [[Paragraph(name, title), Paragraph(contact, contact_style)]],
            colWidths=[112 * mm, 60 * mm],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        ),
        Paragraph(headline, role),
        Spacer(1, 3 * mm),
        HRFlowable(width="100%", thickness=1.2, color=HexColor("#2D6A9F")),
    ]
    for section, entries in sections:
        story.append(Paragraph(section, heading))
        for label, text in entries:
            story.append(Paragraph(label, entry_title))
            story.append(Paragraph(text, body))
            story.append(Spacer(1, 2 * mm))
    doc.build(story)


def main() -> None:
    build_cv(
        "RBA-020-source.pdf",
        name="Elena Testa",
        headline="Data Analyst",
        contact="elena.testa@example.invalid<br/>Bologna, Italy",
        sections=[
            (
                "WORK EXPERIENCE",
                [
                    (
                        "Data Analyst | Meridian Test Cooperative | 2021-04 - Present",
                        "Built reproducible reporting models and maintained quality checks for fictional operations data.",
                    ),
                    (
                        "Reporting Assistant | Sample Civic Lab | 2019-01 - 2021-03",
                        "Prepared weekly service dashboards and documented data definitions.",
                    ),
                ],
            ),
            (
                "EDUCATION AND TRAINING",
                [("MSc Statistics | University of Bologna | 2017 - 2019", "Applied statistics and data visualisation.")],
            ),
            (
                "DIGITAL SKILLS",
                [("Skills", "SQL, Python, Tableau, dimensional modelling")],
            ),
            (
                "LANGUAGE SKILLS",
                [("Italian", "Native"), ("English", "C1")],
            ),
        ],
    )
    build_cv(
        "RBA-021-source.pdf",
        name="Noah Example",
        headline="Accessibility Engineer",
        contact="noah.example@example.invalid<br/>Dublin, Ireland",
        sections=[
            (
                "WORK EXPERIENCE",
                [
                    (
                        "Accessibility Engineer | North Test Studio | 2022-02 - Present",
                        "Runs assistive-technology audits and maintains accessible component guidance.",
                    )
                ],
            ),
            (
                "EDUCATION AND TRAINING",
                [("BEng Software Engineering | Example Institute | 2017 - 2021", "Human-computer interaction pathway.")],
            ),
            (
                "CONFERENCES AND SEMINARS",
                [("Inclusive Interfaces Forum | 2024", "Presented a fictional audit case study.")],
            ),
            (
                "DIGITAL SKILLS",
                [("Skills", "WCAG auditing, HTML, CSS, screen-reader testing")],
            ),
            (
                "LANGUAGE SKILLS",
                [("English", "Native"), ("Irish", "B1")],
            ),
        ],
    )


if __name__ == "__main__":
    main()

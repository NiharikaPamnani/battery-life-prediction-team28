"""Build the assignment PDF from FINAL_REPORT.md and its locally generated figures."""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = ROOT / "docs" / "FINAL_REPORT.md"
PDF_PATH = ROOT / "docs" / "Battery_Life_Prediction_Edge_AI_Report.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportBody", parent=base["BodyText"], fontName="Helvetica",
        fontSize=9, leading=13, spaceAfter=7, textColor=colors.HexColor("#1c2633"),
    ))
    base.add(ParagraphStyle(
        name="ReportH1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=colors.HexColor("#12304a"),
        spaceBefore=13, spaceAfter=7, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ReportH2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=colors.HexColor("#176b68"),
        spaceBefore=9, spaceAfter=5, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ReportH3", parent=base["Heading3"], fontName="Helvetica-Bold",
        fontSize=10, leading=13, textColor=colors.HexColor("#253b59"),
        spaceBefore=7, spaceAfter=4, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ReportCode", fontName="Courier", fontSize=7.2, leading=9.4,
        leftIndent=6, rightIndent=6, borderColor=colors.HexColor("#dbe3ec"),
        borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#f4f7fa"),
        spaceBefore=3, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="ReportCaption", parent=base["BodyText"], fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=colors.HexColor("#596a7c"), spaceAfter=9,
    ))
    base.add(ParagraphStyle(
        name="ReportTable", parent=base["BodyText"], fontSize=7.1, leading=9,
        spaceAfter=0, allowWidows=0, allowOrphans=0,
    ))
    base.add(ParagraphStyle(
        name="ReportCoverTitle", fontName="Helvetica-Bold", fontSize=26,
        leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#102c47"),
    ))
    base.add(ParagraphStyle(
        name="ReportCoverSub", fontName="Helvetica", fontSize=13,
        leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#28746f"),
    ))
    base.add(ParagraphStyle(
        name="TOCHeading1", fontName="Helvetica-Bold", fontSize=9.5,
        leading=13, leftIndent=0, firstLineIndent=0, spaceBefore=3,
    ))
    base.add(ParagraphStyle(
        name="TOCHeading2", fontName="Helvetica", fontSize=8.5,
        leading=11, leftIndent=14, firstLineIndent=0,
    ))
    return base


def inline_markup(value: str) -> str:
    text = html.escape(value.strip())
    code_spans = []

    def preserve_code(match):
        code_spans.append(match.group(1))
        return f"CODESPANPLACEHOLDER{len(code_spans) - 1}END"

    text = re.sub(r"`([^`]+)`", preserve_code, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#176b68">\1</link>', text)
    for index, code in enumerate(code_spans):
        text = text.replace(
            f"CODESPANPLACEHOLDER{index}END",
            f'<font name="Courier">{code}</font>',
        )
    return text


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        level = {"ReportH1": 0, "ReportH2": 1, "ReportH3": 1}.get(style_name)
        if level is None:
            return
        title = flowable.getPlainText()
        key = f"toc-{self.page}-{abs(hash((title, self.page)))}"
        self.canv.bookmarkPage(key)
        self.notify("TOCEntry", (level, title, self.page, key))


def draw_page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d6e0e9"))
    canvas.line(MARGIN, 13 * mm, PAGE_WIDTH - MARGIN, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#617184"))
    canvas.drawString(MARGIN, 8.5 * mm, "Battery Life Prediction using Edge AI")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def parse_markdown(text: str, report_styles) -> list:
    lines = text.splitlines()
    story = []
    paragraph_lines = []
    index = 0
    in_code = False
    code_lines = []
    seen_content_heading = False

    def flush_paragraph():
        if paragraph_lines:
            story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), report_styles["ReportBody"]))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), report_styles["ReportCode"]))
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if line.startswith("## "):
            seen_content_heading = True
        if not seen_content_heading:
            index += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            flush_paragraph()
            alt, reference = image_match.groups()
            image_path = (MARKDOWN_PATH.parent / reference).resolve()
            if image_path.is_file():
                image = Image(str(image_path))
                scale = min(CONTENT_WIDTH / image.imageWidth, 150 * mm / image.imageHeight, 1)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                story.extend([image, Paragraph(inline_markup(alt), report_styles["ReportCaption"])])
            else:
                story.append(Paragraph(f"Figure unavailable: {inline_markup(alt)}", report_styles["ReportCaption"]))
            index += 1
            continue

        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[index + 1]):
            flush_paragraph()
            rows = [table_cells(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(table_cells(lines[index]))
                index += 1
            table_data = [
                [Paragraph(inline_markup(cell), report_styles["ReportTable"]) for cell in row]
                for row in rows
            ]
            column_count = max(map(len, table_data))
            widths = [CONTENT_WIDTH / column_count] * column_count
            table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcebe9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12304a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 7)])
            continue

        heading = re.match(r"^(#{2,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            style = report_styles["ReportH1" if level == 2 else "ReportH2" if level == 3 else "ReportH3"]
            story.append(Paragraph(inline_markup(heading.group(2)), style))
            index += 1
            continue

        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line):
            flush_paragraph()
            items = []
            ordered = bool(re.match(r"^\s*\d+[.)]\s+", line))
            pattern = r"^\s*\d+[.)]\s+" if ordered else r"^\s*[-*]\s+"
            while index < len(lines) and re.match(pattern, lines[index]):
                item_text = re.sub(pattern, "", lines[index]).strip()
                items.append(ListItem(Paragraph(inline_markup(item_text), report_styles["ReportBody"])))
                index += 1
            story.append(ListFlowable(
                items, bulletType="1" if ordered else "bullet", start="1",
                leftIndent=18, bulletFontName="Helvetica", bulletFontSize=8,
            ))
            story.append(Spacer(1, 4))
            continue

        if line.strip() == "---":
            flush_paragraph()
            story.append(Spacer(1, 6))
            index += 1
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line.strip())
        index += 1

    flush_paragraph()
    return story


def main() -> None:
    if not MARKDOWN_PATH.is_file():
        raise FileNotFoundError(f"Report source not found: {MARKDOWN_PATH}")
    report_styles = styles()
    story = [
        Spacer(1, 46 * mm),
        Paragraph("Battery Life Prediction<br/>using Edge AI", report_styles["ReportCoverTitle"]),
        Spacer(1, 9 * mm),
        Paragraph("AI255TBD Edge AI · Module 1", report_styles["ReportCoverSub"]),
        Spacer(1, 4 * mm),
        Paragraph("Team 28 · 22 November 2026", report_styles["ReportCoverSub"]),
        Spacer(1, 65 * mm),
        Paragraph("Neural networks from training through API deployment and edge feasibility.", report_styles["ReportBody"]),
        PageBreak(),
        Paragraph("Table of Contents", report_styles["ReportH1"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [report_styles["TOCHeading1"], report_styles["TOCHeading2"]]
    story.extend([toc, PageBreak()])
    story.extend(parse_markdown(MARKDOWN_PATH.read_text(encoding="utf-8"), report_styles))

    document = ReportDocument(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title="Battery Life Prediction using Edge AI",
        author="Team 28",
    )
    document.multiBuild(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(f"Generated {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

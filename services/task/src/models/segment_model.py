from bs4 import BeautifulSoup
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

from src.models.ocr_model import OCRResult, BoundingBox


class SegmentType(str, Enum):
    Title = "Title"
    SectionHeader = "Section header"
    Text = "Text"
    ListItem = "List item"
    Table = "Table"
    Picture = "Picture"
    Caption = "Caption"
    Formula = "Formula"
    Footnote = "Footnote"
    PageHeader = "Page header"
    PageFooter = "Page footer"


class BaseSegment(BaseModel):
    segment_id: str
    left: float
    top: float
    width: float
    height: float
    text: str
    segment_type: SegmentType
    page_number: int
    page_width: float
    page_height: float


class Segment(BaseModel):
    segment_id: str
    bbox: BoundingBox
    page_number: int
    page_width: float
    page_height: float
    text_layer: str
    ocr_text: Optional[str] = None
    segment_type: SegmentType
    ocr: Optional[List[OCRResult]] = None
    image: Optional[str] = None
    html: Optional[str] = Field(
        None, description="HTML representation of the segment")
    markdown: Optional[str] = Field(
        None, description="Markdown representation of the segment")

    @classmethod
    def from_base_segment(cls, base_segment: BaseSegment) -> 'Segment':
        """
        Create a Segment instance from a BaseSegment instance.
        """
        bbox = BoundingBox(
            top_left=[base_segment.left, base_segment.top],
            top_right=[base_segment.left + base_segment.width, base_segment.top],
            bottom_right=[base_segment.left + base_segment.width,
                          base_segment.top + base_segment.height],
            bottom_left=[base_segment.left, base_segment.top + base_segment.height]
        )

        return cls(
            segment_id=base_segment.segment_id,
            bbox=bbox,
            page_number=base_segment.page_number,
            page_width=base_segment.page_width,
            page_height=base_segment.page_height,
            text=base_segment.text,
            segment_type=base_segment.segment_type
        )

    def _get_content(self):
        if self.ocr:
            return " ".join([result.text for result in self.ocr])
        elif self.text_layer:
            return self.text_layer
        else:
            return ""

    # todo: review weather to sync for formula and latex
    def create_ocr_text(self):
        """
        Generate text representation of the segment based on its type.
        """
        if not self.ocr:
            return

        self.ocr_text = " ".join([result.text for result in self.ocr])

    def create_html(self):
        """
        Extract text from OCR results or use the text field,
        apply HTML formatting based on segment type, and update the html field.
        """
        content = self._get_content()
        if not content:
            return

        if self.segment_type == SegmentType.Title:
            self.html = f"<h1>{content}</h1>"
        elif self.segment_type == SegmentType.SectionHeader:
            self.html = f"<h2>{content}</h2>"
        elif self.segment_type == SegmentType.ListItem:
            self.html = f"<li>{content}</li>"
        elif self.segment_type == SegmentType.Text:
            self.html = f"<p>{content}</p>"
        elif self.segment_type == SegmentType.Picture:
            self.html = f"<img>"
        elif self.segment_type == SegmentType.Table:
            if self.ocr:
                # Already in HTML
                pass
            else:
                rows = content.split('\n')
                table_html = "<table>"
                for row in rows:
                    cells = row.split()
                    table_html += "<tr>" + \
                        "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
                table_html += "</table>"
                self.html = table_html
        else:
            self.html = f'<span class="{self.segment_type.value.lower().replace(" ", "-")}">{content}</span>'

    def create_markdown(self):
        """
        Generate markdown representation of the segment based on its type.
        """
        content = self._get_content()
        if not content:
            return

        if self.segment_type == SegmentType.Title:
            self.markdown = f"# {content}\n\n"
        elif self.segment_type == SegmentType.SectionHeader:
            self.markdown = f"## {content}\n\n"
        elif self.segment_type == SegmentType.ListItem:
            self.markdown = f"- {content}\n"
        elif self.segment_type == SegmentType.Text:
            self.markdown = f"{content}\n\n"
        elif self.segment_type == SegmentType.Picture:
            self.markdown = f"![Image]()\n\n" if self.image else ""
        elif self.segment_type == SegmentType.Table:
            if self.html and self.html.startswith("<table>"):
                self.markdown = self._html_table_to_markdown()
            else:
                # Fallback to simple table representation
                rows = content.split('\n')
                header = "| " + " | ".join(rows[0].split()) + " |"
                separator = "|" + \
                    "|".join(["---" for _ in rows[0].split()]) + "|"
                body = "\n".join("| " + " | ".join(row.split()
                                                   ) + " |" for row in rows[1:])
                self.markdown = f"{header}\n{separator}\n{body}\n\n"
        else:
            self.markdown = f"*{content}*\n\n"

    def _html_table_to_markdown(self):
        """
        Convert HTML table to markdown format.
        """
        soup = BeautifulSoup(self.html, 'html.parser')
        table = soup.find('table')
        markdown_table = []

        header = table.find('tr')
        header_cells = [cell.get_text(strip=True)
                        for cell in header.find_all(['th', 'td'])]
        markdown_table.append("| " + " | ".join(header_cells) + " |")
        markdown_table.append(
            "|" + "|".join(["---" for _ in header_cells]) + "|")

        for row in table.find_all('tr')[1:]:
            cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
            markdown_table.append("| " + " | ".join(cells) + " |")

        return "\n".join(markdown_table) + "\n\n"

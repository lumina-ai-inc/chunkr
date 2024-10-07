from enum import Enum
from markdownify import markdownify as md
from pydantic import BaseModel, Field
from typing import Optional, List

from src.models.ocr_model import OCRResult, BoundingBox
from src.configs.task_config import TASK__OCR_CONFIDENCE_THRESHOLD


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

class Segment(BaseModel):
    segment_id: str
    bbox: BoundingBox
    page_number: int
    page_width: float
    page_height: float
    segment_type: SegmentType
    content: str
    ocr: Optional[List[OCRResult]] = None
    image: Optional[str] = None
    html: Optional[str] = Field(
        None, description="HTML representation of the segment")
    markdown: Optional[str] = Field(
        None, description="Markdown representation of the segment")

    def calculate_avg_ocr_confidence(self):
        if self.ocr:
            avg_confidence = sum(
                result.confidence for result in self.ocr if result.confidence is not None) / len(self.ocr)
            return avg_confidence
        return None

    def _get_content(self):
        if self.ocr:
            avg_confidence = self.calculate_avg_ocr_confidence()
            ocr_text = " ".join([result.text for result in self.ocr])
            if avg_confidence >= TASK__OCR_CONFIDENCE_THRESHOLD:
                return ocr_text
            elif not self.content:
                return ocr_text
        return self.content if self.content else ""

    def _create_content(self):
        """
        Generate text representation of the segment
        """
        self.content = self._get_content()

    def _create_html(self):
        """
        Apply HTML formatting based on segment type, and update the html field.
        """
        content = self.content
        if not content:
            return

        if self.segment_type == SegmentType.Title:
            self.html = f"<h1>{content}</h1>"
        elif self.segment_type == SegmentType.SectionHeader:
            self.html = f"<h2>{content}</h2>"
        elif self.segment_type == SegmentType.ListItem:
            parts = content.strip().split('.')
            if parts[0].isdigit():
                start_number = int(parts[0])
                item = '.'.join(parts[1:]).strip()
                self.html = f"<ol start='{start_number}'><li>{item}</li></ol>"
            else:
                cleaned_content = content.lstrip('-*•● ').strip()
                self.html = f"<ul><li>{cleaned_content}</li></ul>"
        elif self.segment_type == SegmentType.Text:
            self.html = f"<p>{content}</p>"
        elif self.segment_type == SegmentType.Picture:
            self.html = "<img>"
        elif self.segment_type == SegmentType.Table:
            if self.html:
                html_content = self.html.strip()
                if html_content.startswith('<html>') and html_content.endswith('</html>'):
                    html_content = html_content[6:-7].strip()
                if html_content.startswith('<body>') and html_content.endswith('</body>'):
                    html_content = html_content[6:-7].strip()
                self.html = html_content
            else:
                self.html = f'<span class="{self.segment_type.value.lower().replace(" ", "-")}">{content}</span>'
        else:
            self.html = f'<span class="{self.segment_type.value.lower().replace(" ", "-")}">{content}</span>'

    def _create_markdown(self):
        """
        Generate markdown representation of the segment based on its type.
        """
        content = self.content
        if not content:
            return

        if self.segment_type == SegmentType.Title:
            self.markdown = f"# {content}\n\n"
        elif self.segment_type == SegmentType.SectionHeader:
            self.markdown = f"## {content}\n\n"
        elif self.segment_type == SegmentType.ListItem:
            self.markdown = md(self.html)
        elif self.segment_type == SegmentType.Text:
            self.markdown = f"{content}\n\n"
        elif self.segment_type == SegmentType.Picture:
            self.markdown = "![Image]()\n\n" if self.image else ""
        elif self.segment_type == SegmentType.Table:
            if self.html:
                self.markdown = md(self.html)
        else:
            self.markdown = f"*{content}*\n\n"

    def finalize(self):
        self._create_content()
        self._create_html()
        self._create_markdown()

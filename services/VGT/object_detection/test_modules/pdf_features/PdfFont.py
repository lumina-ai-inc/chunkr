from lxml.etree import ElementBase


class PdfFont:
    def __init__(self, font_id: str, bold: bool, italics: bool, font_size: float, color: str):
        self.font_size = font_size
        self.font_id = font_id
        self.bold: bool = bold
        self.italics: bool = italics
        self.color = color

    @staticmethod
    def from_poppler_etree(xml_text_style_tag: ElementBase):
        bold: bool = "Bold" in xml_text_style_tag.attrib["family"]
        italics: bool = "Italic" in xml_text_style_tag.attrib["family"]
        font_size: float = float(xml_text_style_tag.attrib["size"])
        color: str = "#000000" if "color" not in xml_text_style_tag.attrib else xml_text_style_tag.attrib["color"]
        return PdfFont(xml_text_style_tag.attrib["id"], bold, italics, font_size, color)

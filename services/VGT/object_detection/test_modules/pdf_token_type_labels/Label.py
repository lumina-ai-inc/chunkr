from pydantic import BaseModel
from lxml.etree import ElementBase

from pdf_features.Rectangle import Rectangle


class Label(BaseModel):
    top: int
    left: int
    width: int
    height: int
    label_type: int
    metadata: str = ""

    def intersection_percentage(self, token_bounding_box: Rectangle):
        label_bounding_box = Rectangle(
            left=self.left, top=self.top, right=self.left + self.width, bottom=self.top + self.height
        )
        return label_bounding_box.get_intersection_percentage(token_bounding_box)

    def get_location_discrepancy(self, token_bounding_box: Rectangle):
        coordinates_discrepancy: int = abs(self.left - token_bounding_box.left) + abs(self.top - token_bounding_box.top)
        size_discrepancy: int = abs(self.height - token_bounding_box.height) + abs(self.width - token_bounding_box.width)
        return coordinates_discrepancy + size_discrepancy

    def area(self):
        return self.width * self.height

    @staticmethod
    def from_rectangle(rectangle: Rectangle, token_type: int):
        return Label(
            top=rectangle.top, left=rectangle.left, width=rectangle.width, height=rectangle.height, label_type=token_type
        )

    @staticmethod
    def from_text_elements(text_elements: list[ElementBase]):
        top = min([int(x.attrib["top"]) for x in text_elements])
        left = min([int(x.attrib["left"]) for x in text_elements])
        bottom = max([int(x.attrib["top"]) + int(x.attrib["height"]) for x in text_elements])
        right = max([int(x.attrib["left"]) + int(x.attrib["width"]) for x in text_elements])

        return Label(top=top, left=left, width=int(right - left), height=int(bottom - top), label_type=0)

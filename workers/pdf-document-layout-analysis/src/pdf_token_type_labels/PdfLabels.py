from pydantic import BaseModel

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.PageLabels import PageLabels


class PdfLabels(BaseModel):
    pages: list[PageLabels] = list()

    def get_label_type(self, page_number: int, token_bounding_box: Rectangle):
        for page in self.pages:
            if page.number != page_number:
                continue

            return page.get_token_type(token_bounding_box)

        return 6

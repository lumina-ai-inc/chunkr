from toc.MergeTwoSegmentsTitles import MergeTwoSegmentsTitles
from toc.TitleFeatures import TitleFeatures
from toc.data.TOCItem import TOCItem
from toc.PdfSegmentation import PdfSegmentation


class TOCExtractor:
    def __init__(self, pdf_segmentation: PdfSegmentation):
        self.pdf_segmentation = pdf_segmentation
        self.titles_features_sorted = MergeTwoSegmentsTitles(self.pdf_segmentation).titles_merged
        self.toc: list[TOCItem] = list()
        self.set_toc()

    def set_toc(self):
        for index, title_features in enumerate(self.titles_features_sorted):
            indentation = self.get_indentation(index, title_features)
            self.toc.append(title_features.to_toc_item(indentation))

    def __str__(self):
        return "\n".join([f'{"  " * x.indentation} * {x.label}' for x in self.toc])

    def get_indentation(self, title_index: int, title_features: TitleFeatures):
        if title_index == 0:
            return 0

        for index in reversed(range(title_index)):
            if self.toc[index].point_closed:
                continue

            if self.same_indentation(self.titles_features_sorted[index], title_features):
                self.close_toc_items(self.toc[index].indentation)
                return self.toc[index].indentation

        return self.toc[title_index - 1].indentation + 1

    def close_toc_items(self, indentation):
        for toc in self.toc:
            if toc.indentation > indentation:
                toc.point_closed = True

    @staticmethod
    def same_indentation(previous_title_features: TitleFeatures, title_features: TitleFeatures):
        if previous_title_features.first_characters in title_features.get_possible_previous_point():
            return True

        if previous_title_features.get_features_toc() == title_features.get_features_toc():
            return True

        return False

    def to_dict(self):
        toc: list[dict[str, any]] = list()

        for toc_item in self.toc:
            toc_element_dict = dict()
            toc_element_dict["indentation"] = toc_item.indentation
            toc_element_dict["label"] = toc_item.label
            rectangle = dict()
            rectangle["left"] = int(toc_item.selection_rectangle.left)
            rectangle["top"] = int(toc_item.selection_rectangle.top)
            rectangle["width"] = int(toc_item.selection_rectangle.width)
            rectangle["height"] = int(toc_item.selection_rectangle.height)
            rectangle["page"] = str(toc_item.selection_rectangle.page_number)
            toc_element_dict["bounding_box"] = rectangle
            toc.append(toc_element_dict)

        return toc

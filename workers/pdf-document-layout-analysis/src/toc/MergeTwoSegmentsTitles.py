from toc.TitleFeatures import TitleFeatures
from toc.PdfSegmentation import PdfSegmentation


class MergeTwoSegmentsTitles:
    def __init__(self, pdf_segmentation: PdfSegmentation):
        self.title_features_list: list[TitleFeatures] = TitleFeatures.from_pdf_segmentation(pdf_segmentation)
        self.titles_merged: list[TitleFeatures] = list()
        self.merge()

    def merge(self):
        index = 0
        while index < len(self.title_features_list):
            if index == len(self.title_features_list) - 1:
                self.titles_merged.append(self.title_features_list[index])
                break

            if not self.should_merge(self.title_features_list[index], self.title_features_list[index + 1]):
                self.titles_merged.append(self.title_features_list[index])
                index += 1
                continue

            self.title_features_list[index + 1] = self.title_features_list[index + 1].append(self.title_features_list[index])
            index += 1

    @staticmethod
    def should_merge(title: TitleFeatures, other_title: TitleFeatures):
        same_page = other_title.pdf_segment.page_number == title.pdf_segment.page_number

        if not same_page:
            return False

        if abs(other_title.top - title.bottom) > 15:
            return False

        if abs(other_title.left - title.right) > 15 or abs(other_title.right - title.left) > 15:
            return False

        if title.first_characters_type in [1, 2, 3] and other_title.first_characters_type in [1, 2, 3]:
            return False

        if title.bullet_points_type and other_title.bullet_points_type:
            return False

        if title.get_features_to_merge() != other_title.get_features_to_merge():
            return False

        return True

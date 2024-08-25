import os
from os.path import join
from pathlib import Path

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.PageLabels import PageLabels
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TaskMistakesType import TaskMistakesType
from pdf_tokens_type_trainer.config import LABELS_FILE_NAME, MISTAKES_RELATIVE_PATH, STATUS_FILE_NAME


class TaskMistakes:
    def __init__(self, pdf_labeled_data_root_path: str, test_id: str, pdf_name: str):
        self.pdf_labeled_data_root_path = pdf_labeled_data_root_path
        self.test_id = test_id
        self.pdf_name = pdf_name
        self.page_labels: list[PageLabels] = list()

    def add(self, page_number: int, rectangle: Rectangle, truth: int, prediction: int | float, metadata: str = ""):
        token_type_label = Label.from_rectangle(rectangle, self.get_token_type(prediction, truth))
        token_type_label.metadata = metadata
        token_type_page = [x for x in self.page_labels if x.number == page_number]

        if not token_type_page:
            self.page_labels.append(PageLabels(number=page_number, labels=[token_type_label]))
        else:
            token_type_page[0].add_label(token_type_label)

    @staticmethod
    def get_token_type(prediction, truth):
        prediction_integer = round(prediction)

        if truth == prediction_integer:
            return TaskMistakesType.CORRECT.get_index()

        if truth == 1 and prediction_integer == 0:
            return TaskMistakesType.MISSING.get_index()

        return TaskMistakesType.WRONG.get_index()

    def save(self):
        token_type_label_path: str = join(self.pdf_labeled_data_root_path, MISTAKES_RELATIVE_PATH)
        token_type_labels_path = Path(join(token_type_label_path, self.test_id, self.pdf_name, LABELS_FILE_NAME))

        os.makedirs(token_type_labels_path.parent, exist_ok=True)

        token_type_labels = PdfLabels(pages=self.page_labels)
        token_type_labels_path.write_text(token_type_labels.model_dump_json())

        if self.all_correct():
            status_path = Path(join(token_type_label_path, self.test_id, self.pdf_name, STATUS_FILE_NAME))
            status_path.write_text("finished")

    def all_correct(self):
        for token_type_page in self.page_labels:
            for token_type_label in token_type_page.labels:
                if token_type_label.label_type == TaskMistakesType.CORRECT.get_index():
                    continue

                return False

        return True

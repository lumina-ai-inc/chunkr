from pdf_features.Rectangle import Rectangle


class Prediction:
    def __init__(self, bounding_box: Rectangle, category_id: int, score: float):
        self.bounding_box: Rectangle = bounding_box
        self.category_id: int = category_id
        self.score: float = score

from doctr.models import ocr_predictor

if __name__ == "__main__":
    predictor = ocr_predictor('fast_base', 'master', pretrained=True, 
                             export_as_straight_boxes=True)

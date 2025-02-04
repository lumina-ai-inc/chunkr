from doctr.models import ocr_predictor

if __name__ == "__main__":
    predictor = ocr_predictor('db_resnet50', 'parseq', pretrained=True, 
                             export_as_straight_boxes=True)

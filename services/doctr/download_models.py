from doctr.models import ocr_predictor

if __name__ == "__main__":
    predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, 
                             export_as_straight_boxes=True).cuda()
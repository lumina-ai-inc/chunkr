from paddleocr import PaddleOCR, PPStructure

PaddleOCR(use_angle_cls=True, lang="en",
          ocr_order_method="tb-xy", show_log=False)
PPStructure(recovery=True, return_ocr_result_in_table=True,
            layout=False, structure_version="PP-StructureV2", show_log=False)

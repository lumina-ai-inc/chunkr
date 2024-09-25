from paddleocr import PaddleOCR, PPStructure


def init_paddle():
    ppocr = PaddleOCR(use_angle_cls=True, lang="en",
                      ocr_order_method="tb-xy", show_log=True)
    pps = PPStructure(recovery=True, return_ocr_result_in_table=True,
                      layout=False, structure_version="PP-StructureV2", show_log=True)
    print("Paddle initialized")
    return ppocr, pps


if __name__ == "__main__":
    try:
        init_paddle()
    except Exception as e:
        print(f"Error: {e}")

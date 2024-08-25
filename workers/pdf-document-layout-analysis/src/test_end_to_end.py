import requests
from unittest import TestCase
from configuration import ROOT_PATH


class TestEndToEnd(TestCase):
    service_url = "http://localhost:5060"

    def test_error_file(self):
        with open(f"{ROOT_PATH}/test_pdfs/error.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(422, results.status_code)

    def test_blank_pdf(self):
        with open(f"{ROOT_PATH}/test_pdfs/blank.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(200, results.status_code)
            self.assertEqual(0, len(results.json()))

    def test_segmentation_some_empty_pages(self):
        with open(f"{ROOT_PATH}/test_pdfs/some_empty_pages.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(200, results.status_code)
            self.assertEqual(2, len(results.json()))

    def test_image_pdfs(self):
        with open(f"{ROOT_PATH}/test_pdfs/image.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(200, results.status_code)

    def test_regular_pdf(self):
        with open(f"{ROOT_PATH}/test_pdfs/regular.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            results_dict = results.json()
            expected_content = "RESOLUCIÓN DE LA CORTE INTERAMERICANA DE DERECHOS HUMANOS DEL 29 DE JULIO DE 1991"
            self.assertEqual(200, results.status_code)
            self.assertEqual(expected_content, results_dict[0]["text"])
            self.assertEqual(157, results_dict[0]["left"])
            self.assertEqual(105, results_dict[0]["top"])
            self.assertEqual(283, results_dict[0]["width"])
            self.assertEqual(36, results_dict[0]["height"])
            self.assertEqual(1, results_dict[0]["page_number"])
            self.assertEqual(595, results_dict[0]["page_width"])
            self.assertEqual(842, results_dict[0]["page_height"])
            self.assertEqual("Section header", results_dict[0]["type"])

    def test_error_file_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/error.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(422, results.status_code)

    def test_blank_pdf_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/blank.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(200, results.status_code)
            self.assertEqual(0, len(results.json()))

    def test_segmentation_some_empty_pages_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/some_empty_pages.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(200, results.status_code)
            self.assertEqual(2, len(results.json()))

    def test_image_pdfs_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/image.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(200, results.status_code)
            self.assertEqual(0, len(results.json()))

    def test_regular_pdf_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/regular.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}
            results = requests.post(f"{self.service_url}", files=files, data=data)
        results_dict = results.json()
        expected_content = "RESOLUCIÓN DE LA CORTE INTERAMERICANA DE DERECHOS HUMANOS"
        self.assertEqual(200, results.status_code)
        self.assertEqual(expected_content, results_dict[0]["text"])
        self.assertEqual(157, results_dict[0]["left"])
        self.assertEqual(106, results_dict[0]["top"])
        self.assertEqual(284, results_dict[0]["width"])
        self.assertEqual(24, results_dict[0]["height"])
        self.assertEqual(1, results_dict[0]["page_number"])
        self.assertEqual(595, results_dict[0]["page_width"])
        self.assertEqual(842, results_dict[0]["page_height"])
        self.assertEqual("Section header", results_dict[0]["type"])

    def test_korean(self):
        with open(f"{ROOT_PATH}/test_pdfs/korean.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(200, results.status_code)

    def test_chinese(self):
        with open(f"{ROOT_PATH}/test_pdfs/chinese.pdf", "rb") as stream:
            files = {"file": stream}

            results = requests.post(f"{self.service_url}", files=files)

            self.assertEqual(200, results.status_code)

    def test_korean_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/korean.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(200, results.status_code)

    def test_chinese_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/chinese.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            results = requests.post(f"{self.service_url}", files=files, data=data)

            self.assertEqual(200, results.status_code)

    def test_toc(self):
        with open(f"{ROOT_PATH}/test_pdfs/toc-test.pdf", "rb") as stream:
            files = {"file": stream}

            response = requests.post(f"{self.service_url}/toc", files=files)

            response_json = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response_json), 5)
            self.assertEqual(response_json[0]["label"], "TEST")
            self.assertEqual(response_json[0]["indentation"], 0)
            self.assertEqual(response_json[-1]["label"], "C. TITLE LONGER")
            self.assertEqual(response_json[-1]["indentation"], 2)

    def test_toc_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/toc-test.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            response = requests.post(f"{self.service_url}/toc", files=files, data=data)

            response_json = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response_json), 5)
            self.assertEqual(response_json[0]["label"], "TEST")
            self.assertEqual(response_json[0]["indentation"], 0)
            self.assertEqual(response_json[-1]["label"], "C. TITLE LONGER")
            self.assertEqual(response_json[-1]["indentation"], 2)

    def test_text_extraction(self):
        with open(f"{ROOT_PATH}/test_pdfs/test.pdf", "rb") as stream:
            files = {"file": stream}

            response = requests.post(f"{self.service_url}/text", files=files)

            response_json = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response_json.split()[0], "Document")
            self.assertEqual(response_json.split()[1], "Big")
            self.assertEqual(response_json.split()[-1], "TEXT")

    def test_text_extraction_fast(self):
        with open(f"{ROOT_PATH}/test_pdfs/test.pdf", "rb") as stream:
            files = {"file": stream}
            data = {"fast": "True"}

            response = requests.post(f"{self.service_url}/text", files=files, data=data)

            response_json = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response_json.split()[0], "Document")
            self.assertEqual(response_json.split()[1], "Big")
            self.assertEqual(response_json.split()[-1], "TEXT")


import unittest
import requests
import os
import json
from pathlib import Path
from annotate import draw_bounding_boxes
import dotenv

dotenv.load_dotenv(override=True)

class TestPDLAServer(unittest.TestCase):
    BASE_URL = os.getenv("PDLASERVER_URL")
    
    def setUp(self):
        self.input_folder = Path(__file__).parent / "input"
        self.output_folder = Path(__file__).parent / "output"
        if not self.input_folder.exists() or not self.input_folder.is_dir():
            raise FileNotFoundError(f"Input folder not found at {self.input_folder}")
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)
        
        self.test_pdfs = list(self.input_folder.glob("*.pdf"))
        if not self.test_pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.input_folder}")

    def test_pdla_high_quality_extraction(self):
        url = f"{self.BASE_URL}/"
        
        for pdf_path in self.test_pdfs:
            with self.subTest(pdf=pdf_path.name):
                print(f"Processing PDF file: {pdf_path}")
                with open(pdf_path, "rb") as pdf_file:
                    files = {"file": (pdf_path.name, pdf_file, "application/pdf")}
                    data = {"fast": "false", "density": 72, "extension": "jpeg"}
                    
                    response = requests.post(url, files=files, data=data)
                
                self.assertEqual(response.status_code, 200)
                
                json_response = response.json()
                self.assertIsInstance(json_response, list)
                self.assertTrue(len(json_response) > 0)
                
                # Check if the response contains expected fields
                first_item = json_response[0]
                expected_fields = ["left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"]
                for field in expected_fields:
                    self.assertIn(field, first_item)
                
                # Save output to JSON file
                output_json_path = self.output_folder / f"{pdf_path.stem}_json.json"
                with open(output_json_path, "w") as json_file:
                    json.dump(json_response, json_file, indent=2)
                
                # Annotate the PDF
                output_pdf_path = self.output_folder / f"{pdf_path.stem}_Annotated.pdf"
                draw_bounding_boxes(str(pdf_path), json_response, str(output_pdf_path))

if __name__ == "__main__":
    unittest.main()


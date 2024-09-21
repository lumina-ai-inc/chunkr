
import unittest
import requests
import os
import json
from pathlib import Path
from annotate import draw_bounding_boxes
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

dotenv.load_dotenv(override=True)

class TestPDLAServer(unittest.TestCase):
    BASE_URL = os.getenv("PDLASERVER_URL")
    MAX_WORKERS = 4  # Adjust this value based on your system's capabilities
    
    def setUp(self):
        print("Setting up test environment...")
        self.input_folder = Path(__file__).parent / "input"
        self.output_folder = Path(__file__).parent / "output"
        if not self.input_folder.exists() or not self.input_folder.is_dir():
            raise FileNotFoundError(f"Input folder not found at {self.input_folder}")
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)
        
        self.test_pdfs = list(self.input_folder.glob("*.pdf"))
        if not self.test_pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.input_folder}")
        print(f"Found {len(self.test_pdfs)} PDF files for testing.")

    def process_pdf(self, pdf_path):
        url = f"{self.BASE_URL}/"
        print(f"Processing PDF file: {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": (pdf_path.name, pdf_file, "application/pdf")}
            data = {"fast": "false", "density": 72, "extension": "jpeg"}
            
            print(f"Sending POST request to PDLA server for {pdf_path.name}...")
            response = requests.post(url, files=files, data=data)
        
        print(f"Response status code for {pdf_path.name}: {response.status_code}")
        self.assertEqual(response.status_code, 200)
        
        json_response = response.json()
        print(f"Received JSON response with {len(json_response)} items for {pdf_path.name}.")
        self.assertIsInstance(json_response, list)
        self.assertTrue(len(json_response) > 0)
        
        # Check if the response contains expected fields
        first_item = json_response[0]
        expected_fields = ["left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"]
        for field in expected_fields:
            print(f"Checking for field: {field} in {pdf_path.name}")
            self.assertIn(field, first_item)
        
        # Save output to JSON file
        output_json_path = self.output_folder / f"{pdf_path.stem}_json.json"
        print(f"Saving JSON output to: {output_json_path}")
        with open(output_json_path, "w") as json_file:
            json.dump(json_response, json_file, indent=2)
        
        # Annotate the PDF
        output_pdf_path = self.output_folder / f"{pdf_path.stem}_Annotated.pdf"
        print(f"Annotating PDF and saving to: {output_pdf_path}")
        draw_bounding_boxes(str(pdf_path), json_response, str(output_pdf_path))
        print(f"Finished processing {pdf_path.name}")

    def test_pdla_high_quality_extraction(self):
        url = f"{self.BASE_URL}/"
        print(f"Testing PDLA high quality extraction at URL: {url}")
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_pdf, pdf_path) for pdf_path in self.test_pdfs]
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions that occurred during execution

if __name__ == "__main__":
    print("Starting PDLA server tests...")
    unittest.main()
    print("PDLA server tests completed.")


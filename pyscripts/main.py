import os
from datetime import datetime
import concurrent.futures
import glob
import time
from enum import Enum
import urllib.request
from api import process_file
from models import Model, OcrStrategy, UploadForm, TaskResponse
from annotate import draw_bounding_boxes
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

# Models for strucctured extraction. 
class Property(BaseModel):
    """Represents each property within the schema."""
    name: str = Field(..., description="Name of the property")
    title: Optional[str] = Field(None, description="Title of the property")
    type: str = Field(..., description="Type of the property (e.g., 'obj' or 'value')")
    description: Optional[str] = Field(None, description="Description of the property")
    default: Optional[str] = Field(None, description="Default value for the property")

class JsonSchema(BaseModel):
    """Represents the structure of the incoming schema."""
    title: str = Field(..., description="Title of the schema")
    type: str = Field(..., description="Type of the schema (e.g., 'object')")
    properties: List[Property] = Field(default_factory=list, description="List of properties in the schema")

async def print_time_taken(created_at, finished_at):
    if created_at and finished_at:
        try:
            start_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(
                finished_at.strip('"').replace(" UTC", "+00:00")
            )
            time_taken = end_time - start_time
            print(f"Time taken: {time_taken}")
        except ValueError:
            print("Unable to calculate time taken due to invalid timestamp format")
    else:
        print("Time taken information not available")

async def save_to_json(file_path: str, output: json, file_name: str ):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    
    # Convert the output to a dictionary if possible
    if hasattr(output, 'dict'):
        output = output.dict()
    
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)  # Added indent for readability
    return output_json_path

async def extract_and_annotate_file(file_path: str, model: Model, target_chunk_length: int = None, ocr_strategy: OcrStrategy = OcrStrategy.Auto, json_schema_serialized = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(file_path).split(".")[0]
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    output_annotated_path = os.path.join(output_dir, f"{file_name}_annotated.pdf")

    print(f"Processing file: {file_path}")
    
    # Create an UploadForm instance with the serialized JSON schema
    upload_form = UploadForm(
        file=file_path,
        model=model,
        target_chunk_length=target_chunk_length,
        ocr_strategy=ocr_strategy,
        json_schema=json_schema_serialized  # Pass the JSON dict
    )
    
    task: TaskResponse = process_file(upload_form)
    output = task.output
    print(f"File processed: {file_path}")
    print("OUTPUT", output.extracted_json)
    if output is None:
        raise Exception(f"Output not found for {file_path}")

    print(f"Downloading bounding boxes for {file_path}...")
    output_json_path = await save_to_json(output_json_path, output, file_name)
    print(f"Downloaded bounding boxes for {file_path}")

    if task.pdf_url:
        temp_pdf_path = os.path.join(output_dir, f"{file_name}_temp.pdf")
        urllib.request.urlretrieve(task.pdf_url, temp_pdf_path)
        print(f"Annotating file: {temp_pdf_path}")
        draw_bounding_boxes(temp_pdf_path, output.chunks, output_annotated_path)
        os.remove(temp_pdf_path)
    else:
        draw_bounding_boxes(file_path, output, output_annotated_path)
    print(f"File annotated: {file_path}")

async def main(model: Model = Model.HighQuality, target_chunk_length: int = None, ocr_strategy: OcrStrategy = OcrStrategy.Auto, dir="input", json_schema: JsonSchema = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_schema_serialized = json_schema.model_dump() if json_schema else None
    input_dir = os.path.join(current_dir, dir)
    input_files = []
    for extension in ["*.pdf", "*.docx", "*.ppt"]:
        input_files.extend(glob.glob(os.path.join(input_dir, extension)))

    if not input_files:
        print("No PDF files found in the input folder.")
        return

    print(f"Processing {len(input_files)} files...")
    elapsed_times = []

    async def timed_extract(file_path):
        start_time = time.time()
        await extract_and_annotate_file(file_path, model, target_chunk_length, ocr_strategy, json_schema_serialized)
        end_time = time.time()
        return {'file_path': file_path, 'elapsed_time': end_time - start_time}
        
    tasks = [timed_extract(file_path) for file_path in input_files]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            print(f"An error occurred: {str(result)}")
        else:
            elapsed_times.append(result)

    print("All files processed.")
    return elapsed_times


if __name__ == "__main__":
    model = Model.Fast
    target_chunk_length = 1000
    ocr_strategy = OcrStrategy.Auto
    json_schema = JsonSchema(
        title="Clinical Trial Results",
        type="object", 
        properties=[
        Property(
            name="the_company",
            title="Company", 
            type="string",
            description="The legal name of the company or corporation that is recieving the funds from the investor",
            default=None
        ),
        Property(
            name="the_investor",
            title="Investor",
            type="string", 
            description="The legal name of the entity that is investing in the company. The investor could be an individual, a corporation, or a government agency.",
            default=None
        ),
        Property(
            name="purchase_amount",
            title="Purchase Amount",
            type="int",
            description="The purchase amount of the investment in USD",
            default=None
        )
        ]
    )
    times = asyncio.run(main( model, target_chunk_length, ocr_strategy, "input", json_schema=json_schema))  
    
    if times:
        total_time = sum(result['elapsed_time'] for result in times)
        print(f"Total time taken to process all files: {total_time:.2f} seconds")
        
        for result in times:
            print(f"Time taken to process {result['file_path']}: {result['elapsed_time']:.2f} seconds")
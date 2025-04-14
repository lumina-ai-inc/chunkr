# main.py
import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from io import BytesIO
import time
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware # To allow frontend requests
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
from pdf2image import convert_from_bytes # Use convert_from_bytes
from PIL import Image
import img2pdf
import json
# Assuming storage_sync.py is in the same directory and updated
from storage_sync import S3DatasetSync

# --- Configuration & Initialization ---
load_dotenv(override=True)

app = FastAPI(title="PDF Annotator API")

# CORS Middleware: Allow requests from your frontend (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your frontend origin e.g., "http://localhost:8000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], # Allow Authorization header
)

# Retrieve the expected API key from environment variables
EXPECTED_API_KEY = os.getenv("SERVER_API_KEY")
if not EXPECTED_API_KEY:
    print("WARNING: SERVER_API_KEY environment variable not set. API will be insecure.")
    # In a real application, you might want to raise an error here or provide a default insecure key
    # raise ValueError("SERVER_API_KEY environment variable must be set")

# Define the API key header scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(key: str = Depends(api_key_header)):
    """Dependency to verify the API key provided in the Authorization header."""
    if not EXPECTED_API_KEY: # If server key isn't set, bypass security (not recommended for prod)
        print("WARNING: Bypassing API key check because SERVER_API_KEY is not set.")
        return True

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Expecting "Bearer <key>" format
    parts = key.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_key = parts[1]
    if provided_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True # Key is valid

# Initialize S3 Sync (ensure AWS env vars and S3_BUCKET are set)
s3_sync = S3DatasetSync()
if not s3_sync.is_enabled():
    print("WARNING: S3 storage is not configured. API functionality will be limited.")
    # You might want to raise an exception or handle this more gracefully

# Initialize OpenAI Client
try:
    openai_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
except Exception as e:
    print(f"WARNING: Failed to initialize OpenAI client: {e}")
    openai_client = None

# --- Helper Functions (Adapted from PDFAnnotator) ---

def _get_temp_pdf_path(dataset_id: str, pdf_filename: str) -> Optional[str]:
    """Downloads PDF from S3 to a temporary file if not already present."""
    # Simple caching mechanism (could be improved)
    temp_path = s3_sync.temp_dir / f"{dataset_id}_{pdf_filename}"
    if temp_path.exists():
        return str(temp_path)

    downloaded_path = s3_sync.download_pdf_to_temp(dataset_id, pdf_filename)
    return downloaded_path

def _extract_page_text(pdf_path: str, page_num: int) -> str:
    """Extract text from a single page of a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if 0 <= page_num < len(reader.pages):
                page = reader.pages[page_num]
                return page.extract_text() or ""
            else:
                return ""
    except Exception as e:
        print(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
        return ""

def _convert_image_to_pdf_bytes(image_bytes):
    """Convert image bytes to PDF bytes."""
    try:
        img_stream = BytesIO(image_bytes)
        img = Image.open(img_stream)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Use img2pdf directly with the Image object
        pdf_bytes = img2pdf.convert([img]) # Pass as a list
        return pdf_bytes
    except Exception as e:
        print(f"Error converting image to PDF bytes: {e}")
        return None

# --- API Endpoints ---

@app.get("/datasets", response_model=List[str])
async def get_datasets(valid: bool = Depends(verify_api_key)):
    """Lists available dataset IDs from S3."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")
    datasets = s3_sync.list_datasets()
    # Ensure default exists if none are found
    if not datasets:
        s3_sync.ensure_dataset_structure("default")
        return ["default"]
    return datasets

@app.post("/datasets/{dataset_id}", status_code=201)
async def create_dataset(dataset_id: str, valid: bool = Depends(verify_api_key)):
    """Creates the necessary S3 structure for a new dataset."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")
    s3_sync.ensure_dataset_structure(dataset_id)
    return {"message": f"Dataset structure ensured for '{dataset_id}'"}

@app.get("/datasets/{dataset_id}/pdfs", response_model=List[str])
async def get_pdf_list(dataset_id: str, only_unannotated: bool = False, valid: bool = Depends(verify_api_key)):
    """Lists PDF filenames in a specific dataset."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    pdf_files = s3_sync.list_pdfs_in_dataset(dataset_id)

    if only_unannotated:
        try:
            content = s3_sync.download_annotation_content(dataset_id)
            annotations = []
            if content:
                annotations = [json.loads(line) for line in content.strip().split('\n') if line]
            annotated_doc_ids = {ann.get("doc_id") for ann in annotations if ann.get("doc_id")}
            pdf_files = [
                pdf for pdf in pdf_files
                if os.path.splitext(pdf)[0] not in annotated_doc_ids
            ]
        except Exception as e:
            print(f"Error filtering unannotated PDFs for {dataset_id}: {e}")
            # Decide whether to return all or raise error
            # raise HTTPException(status_code=500, detail="Failed to filter unannotated PDFs")

    return pdf_files

@app.post("/datasets/{dataset_id}/pdfs", status_code=201)
async def upload_pdf_endpoint(dataset_id: str, file: UploadFile = File(...), valid: bool = Depends(verify_api_key)):
    """Uploads a PDF or image (converted to PDF) to the dataset in S3."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")
    if not file.filename:
         raise HTTPException(status_code=400, detail="No filename provided")

    file_uuid = str(uuid.uuid4())
    original_name = file.filename
    file_extension = os.path.splitext(original_name)[1].lower()
    new_filename = f"{file_uuid}.pdf" # Always save as PDF
    s3_key = f"{s3_sync._get_pdf_prefix(dataset_id)}{new_filename}"

    try:
        file_bytes = await file.read()
        upload_success = False

        if file_extension in ['.jpg', '.jpeg', '.png']:
            pdf_bytes = _convert_image_to_pdf_bytes(file_bytes)
            if pdf_bytes:
                upload_success = s3_sync.upload_bytes(pdf_bytes, s3_key, content_type='application/pdf')
            else:
                 raise HTTPException(status_code=500, detail=f"Failed to convert image {original_name} to PDF")
        elif file_extension == '.pdf':
            upload_success = s3_sync.upload_bytes(file_bytes, s3_key, content_type='application/pdf')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        if upload_success:
            return {
                 "message": "File uploaded successfully",
                 "original_name": original_name,
                 "saved_as": new_filename,
                 "s3_key": s3_key
             }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to upload {original_name} to S3")
    except HTTPException as he:
        raise he # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error uploading file {original_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {e}")
    finally:
        await file.close()


@app.get("/datasets/{dataset_id}/pdfs/{pdf_filename}")
async def get_pdf_file(dataset_id: str, pdf_filename: str, valid: bool = Depends(verify_api_key)):
    """Downloads the PDF from S3 and serves it."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    temp_pdf_path = _get_temp_pdf_path(dataset_id, pdf_filename)
    if not temp_pdf_path or not Path(temp_pdf_path).exists():
        raise HTTPException(status_code=404, detail=f"PDF '{pdf_filename}' not found or download failed.")

    # Return the file directly
    # Consider adding headers like Content-Disposition if needed
    return FileResponse(temp_pdf_path, media_type='application/pdf', filename=pdf_filename)

@app.get("/datasets/{dataset_id}/annotations", response_model=List[Dict])
async def get_annotations_endpoint(dataset_id: str, doc_id: Optional[str] = None, valid: bool = Depends(verify_api_key)):
    """Gets all annotations for a dataset or optionally filters by doc_id."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    content = s3_sync.download_annotation_content(dataset_id)
    if content is None: # Indicates download error
        raise HTTPException(status_code=500, detail="Failed to download annotations from S3")

    annotations = []
    if content:
        try:
            annotations = [json.loads(line) for line in content.strip().split('\n') if line]
        except json.JSONDecodeError as e:
             raise HTTPException(status_code=500, detail=f"Error decoding annotation JSON: {e}")

    if doc_id:
        annotations = [ann for ann in annotations if ann.get("doc_id") == doc_id]

    return annotations

@app.post("/datasets/{dataset_id}/annotations", status_code=201)
async def save_annotation_endpoint(dataset_id: str, annotation_data: Dict = Body(...), valid: bool = Depends(verify_api_key)):
    """Saves a new annotation to the dataset's JSONL file in S3."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    # Add debug logs
    print(f"Received annotation data: {annotation_data}")

    # Basic validation (can be enhanced with Pydantic models)
    required_keys = ["question", "answers", "doc_id", "page_ids"]
    if not all(key in annotation_data for key in required_keys):
        missing_keys = [key for key in required_keys if key not in annotation_data]
        error_detail = f"Missing required keys: {missing_keys}"
        print(error_detail)
        raise HTTPException(status_code=400, detail=error_detail)

    # --- Read-Modify-Write ---
    # Note: This is NOT safe for concurrent writes. Consider locking or a database.
    try:
        # 1. Read existing content
        content = s3_sync.download_annotation_content(dataset_id)
        if content is None:
            print(f"No existing annotations for dataset {dataset_id}")
            content = ""  # Start with empty content if no file exists

        # 2. Prepare new annotation line
        # Add timestamp, answer_page_idx etc. as needed
        annotation_data["annotation_id"] = str(uuid.uuid4())  # Add unique ID
        annotation_data["questionId"] = int(time.time() * 1000)  # Example ID
        
        if not annotation_data.get("answer_page_idx") and annotation_data["page_ids"]:
            try:
                first_page_id = annotation_data["page_ids"][0]
                page_suffix = first_page_id.split('_p')[-1]
                annotation_data["answer_page_idx"] = int(page_suffix)
            except Exception as e:
                print(f"Error parsing page_id: {e}")
                annotation_data["answer_page_idx"] = 0
        else:
            annotation_data["answer_page_idx"] = 0

        # Make sure data_split exists
        if "data_split" not in annotation_data:
            annotation_data["data_split"] = "test"

        new_line = json.dumps(annotation_data)
        print(f"Prepared new annotation line: {new_line}")

        # 3. Append and Write back
        updated_content = content.strip() + "\n" + new_line if content.strip() else new_line
        success = s3_sync.upload_annotation_content(dataset_id, updated_content)

        if success:
            print(f"Successfully saved annotation for dataset {dataset_id}")
            return annotation_data  # Return the saved annotation
        else:
            print(f"Failed to upload updated annotations to S3 for dataset {dataset_id}")
            raise HTTPException(status_code=500, detail="Failed to upload updated annotations to S3")

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error saving annotation for {dataset_id}: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/datasets/{dataset_id}/pdfs/{pdf_filename}/extract-text", response_model=Dict[int, str])
async def extract_text_endpoint(dataset_id: str, pdf_filename: str, pages: List[int] = Body(...), valid: bool = Depends(verify_api_key)):
    """Extracts text from specified pages of a PDF."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    temp_pdf_path = _get_temp_pdf_path(dataset_id, pdf_filename)
    if not temp_pdf_path:
        raise HTTPException(status_code=404, detail=f"PDF '{pdf_filename}' not found or download failed.")

    extracted_texts = {}
    try:
        # Check page count first
        with open(temp_pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

        for page_num in pages:
            if 0 <= page_num < num_pages:
                extracted_texts[page_num] = _extract_page_text(temp_pdf_path, page_num)
            else:
                extracted_texts[page_num] = f"Error: Page {page_num} out of bounds (0-{num_pages-1})"
        return extracted_texts
    except Exception as e:
        print(f"Error during text extraction for {pdf_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")


@app.post("/datasets/{dataset_id}/pdfs/{pdf_filename}/generate-qa")
async def generate_qa_endpoint(
    dataset_id: str, 
    pdf_filename: str, 
    data: Dict[str, Any] = Body(...),
    valid: bool = Depends(verify_api_key)
):
    """Generates QA suggestions for specified pages using OpenAI."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not configured")
    
    # pages are received as 0-based indices
    pages = data.get("pages", [])
    num_questions = data.get("num_questions", 3)
    
    if not pages:
        raise HTTPException(status_code=400, detail="No pages specified")

    temp_pdf_path = _get_temp_pdf_path(dataset_id, pdf_filename)
    if not temp_pdf_path:
        raise HTTPException(status_code=404, detail=f"PDF '{pdf_filename}' not found")

    try:
        # Convert PDF pages to images
        images = []
        with open(temp_pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            pdf_images = convert_from_bytes(pdf_bytes)
            for page_num in pages:  # page_num is 0-based
                if 0 <= page_num < len(pdf_images):
                    images.append(pdf_images[page_num])
                else:
                    print(f"Skipping page {page_num} (out of bounds)")

        if not images:
            return {"suggestions": []}

        # Following approach from annotator.py
        messages = []
        
        # System prompt
        messages.append({
            "role": "system",
            "content": "You are an expert at creating high-quality question-answer pairs from document content. Generate exactly the requested number of questions."
        })
        
        # Convert 0-based pages to 1-based for human-readable prompt
        one_based_pages = [p + 1 for p in pages]
        prompt_text = f"""Generate exactly {num_questions} question-answer pairs based on the content visible in the selected pages {one_based_pages}.
        Page numbers are in order and start from 1.
        Try to ask visual questions based on the pages you see.
        Requirements:
        1. Questions must be specific and answerable from the visible content
        2. Answers should be concise but complete
        3. Include source page numbers for each answer
        4. Format as JSON array: [{{"question": "...", "answer": "...", "source_pages": [...]}}]
        5. Generate exactly {num_questions} pairs, no more, no less
        
        Return the json with ```json ``` tags."""
        
        # Initial message with text instructions
        messages.append({"role": "user", "content": prompt_text})
        
        # Add image content to messages
        for i, img in enumerate(images):
            # Convert PIL image to bytes
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Encode as base64
            import base64
            base64_img = base64.b64encode(img_bytes).decode('utf-8')
            
            # Use 1-based page numbers in the prompt
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"This is page {pages[i] + 1}. Use this content to generate Q&A pairs."
                    }
                ]
            })

        response = openai_client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=messages,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON with more resilient approach
        try:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(1))
            else:
                # Try to find JSON array directly if not in markdown
                json_match_direct = re.search(r'(\[.*?\])', content, re.DOTALL)
                if json_match_direct:
                    suggestions = json.loads(json_match_direct.group(1))
                else:
                    suggestions = json.loads(content)  # Try parsing directly as a fallback
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM response: {content[:200]}...")
            return {"suggestions": []}
            
        # Validate structure and convert 1-based page numbers from LLM to 0-based
        valid_suggestions = []
        for s in suggestions:
            if isinstance(s, dict) and "question" in s and "answer" in s:
                # Convert 1-based source_pages from LLM to 0-based
                source_pages = s.get("source_pages", [pages[0] + 1])  # Default to first page (1-based)
                if not isinstance(source_pages, list):
                    source_pages = [pages[0] + 1]
                
                # Convert to 0-based and validate
                zero_based_pages = [p - 1 for p in source_pages if isinstance(p, int)]
                # Filter pages to only those from original selection
                zero_based_pages = [p for p in zero_based_pages if p in pages]
                if not zero_based_pages:
                    zero_based_pages = [pages[0]]  # Default to first selected page
                    
                valid_suggestions.append({
                    "question": s["question"],
                    "answer": s["answer"],
                    "source_pages": zero_based_pages  # Store as 0-based
                })
        
        return {"suggestions": valid_suggestions[:num_questions]}
            
    except Exception as e:
        print(f"Error generating QA: {str(e)}")
        return {"suggestions": []}

@app.delete("/datasets/{dataset_id}/annotations/{annotation_id}")
async def delete_annotation_endpoint(dataset_id: str, annotation_id: str, valid: bool = Depends(verify_api_key)):
    """Deletes a specific annotation from the dataset."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    try:
        # Read existing content
        content = s3_sync.download_annotation_content(dataset_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Annotations file not found")

        # Parse and filter out the annotation to delete
        annotations = [json.loads(line) for line in content.strip().split('\n') if line]
        filtered_annotations = [ann for ann in annotations if ann.get("annotation_id") != annotation_id]

        if len(filtered_annotations) == len(annotations):
            raise HTTPException(status_code=404, detail="Annotation not found")

        # Write back filtered content
        updated_content = "\n".join(json.dumps(ann) for ann in filtered_annotations)
        success = s3_sync.upload_annotation_content(dataset_id, updated_content)

        if success:
            return {"message": "Annotation deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update annotations file")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting annotation: {str(e)}")

@app.delete("/datasets/{dataset_id}/pdfs/{pdf_filename}")
async def delete_pdf_endpoint(dataset_id: str, pdf_filename: str, valid: bool = Depends(verify_api_key)):
    """Deletes a PDF file from the dataset."""
    if not s3_sync.is_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    try:
        # First check if the PDF exists
        pdf_files = s3_sync.list_pdfs_in_dataset(dataset_id)
        if pdf_filename not in pdf_files:
            raise HTTPException(status_code=404, detail="PDF not found")

        # Get the full S3 key for the PDF
        s3_key = f"{s3_sync._get_pdf_prefix(dataset_id)}{pdf_filename}"
        
        # Delete the PDF file from S3 - Fixed to use s3_bucket instead of bucket_name
        success = s3_sync.s3_client.delete_object(
            Bucket=s3_sync.s3_bucket,
            Key=s3_key
        )

        # Also delete the local temp file if it exists
        temp_path = s3_sync.temp_dir / f"{dataset_id}_{pdf_filename}"
        if temp_path.exists():
            temp_path.unlink()

        # Delete any annotations for this PDF
        doc_id = os.path.splitext(pdf_filename)[0]
        content = s3_sync.download_annotation_content(dataset_id)
        if content:
            annotations = [json.loads(line) for line in content.strip().split('\n') if line]
            filtered_annotations = [ann for ann in annotations if ann.get("doc_id") != doc_id]
            
            if len(filtered_annotations) != len(annotations):
                updated_content = "\n".join(json.dumps(ann) for ann in filtered_annotations)
                s3_sync.upload_annotation_content(dataset_id, updated_content)

        return {"message": "PDF and related annotations deleted successfully"}

    except Exception as e:
        print(f"Error deleting PDF {pdf_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

# --- Optional: Serve Static Frontend Files ---
# If you want FastAPI to serve your HTML/CSS/JS
# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")
# @app.get("/")
# async def read_index():
#     return FileResponse('static/index.html')

# --- Run the server (for development) ---
if __name__ == "__main__":
    # Clean up temp dir on startup (optional)
    # try:
    #     s3_sync.cleanup_temp_dir()
    # except Exception as e:
    #     print(f"Error cleaning temp dir on startup: {e}")

    uvicorn.run(app, host="0.0.0.0", port=6969) # Use a different port than Streamlit if running simultaneously
import streamlit as st
import os
import json
import base64
import time
import uuid
from pathlib import Path
import PyPDF2
from io import BytesIO
import shutil
import requests
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
import random
from storage_sync import S3DatasetSync
import threading
from datetime import datetime
import dotenv
import tempfile # For temporary file handling

def setup_aws_config():
    """Set up AWS configuration with UI if env vars aren't set."""
    st.sidebar.subheader("AWS S3 Configuration")
    # Load environment variables from .env file
    load_dotenv(override=True)
    
    # Check if AWS credentials are already set in environment
    has_aws_config = all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("S3_BUCKET")
    ])
    
    if has_aws_config:
        st.sidebar.success("AWS credentials already configured via environment variables")
        s3_bucket = os.getenv("S3_BUCKET")
        st.sidebar.info(f"Using S3 bucket: {s3_bucket}")
        return True
    
    # Option to configure via UI
    with st.sidebar.expander("Configure AWS Credentials", expanded=not has_aws_config):
        st.warning("AWS credentials not found in environment variables")
        st.info("Enter your AWS credentials to enable S3 sync")
        
        # Use session state to store credentials during the session
        if "aws_access_key" not in st.session_state:
            st.session_state.aws_access_key = ""
        if "aws_secret_key" not in st.session_state:
            st.session_state.aws_secret_key = ""
        if "s3_bucket" not in st.session_state:
            st.session_state.s3_bucket = ""
        if "aws_region" not in st.session_state:
            st.session_state.aws_region = "us-east-1"  # Default region
        
        # Credential input fields
        aws_access_key = st.text_input(
            "AWS Access Key ID", 
            value=st.session_state.aws_access_key,
            type="password",
            help="Your AWS Access Key ID"
        )
        aws_secret_key = st.text_input(
            "AWS Secret Access Key", 
            value=st.session_state.aws_secret_key,
            type="password",
            help="Your AWS Secret Access Key"
        )
        s3_bucket = st.text_input(
            "S3 Bucket Name", 
            value=st.session_state.s3_bucket,
            help="Name of the S3 bucket for dataset storage"
        )
        aws_region = st.text_input(
            "AWS Region (optional)", 
            value=st.session_state.aws_region,
            help="AWS region, e.g., us-east-1"
        )
        
        if st.button("Save AWS Configuration"):
            if not aws_access_key or not aws_secret_key or not s3_bucket:
                st.error("Please fill all required fields")
                return False
                
            # Store in session state
            st.session_state.aws_access_key = aws_access_key
            st.session_state.aws_secret_key = aws_secret_key
            st.session_state.s3_bucket = s3_bucket
            st.session_state.aws_region = aws_region
            
            # Set as environment variables for current session
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
            os.environ["S3_BUCKET"] = s3_bucket
            if aws_region:
                os.environ["AWS_REGION"] = aws_region
                
            st.success("AWS credentials saved for this session")
            st.info("Restart required to apply changes. Click 'Restart Application'")
            if st.button("Restart Application"):
                st.rerun()
            return True
            
    # Option to load from .env file
    with st.sidebar.expander("Load from .env file"):
        env_file = st.file_uploader(
            ".env file", 
            type=["env", "txt"],
            help="Upload a .env file containing AWS credentials"
        )
        
        if env_file and st.button("Load .env file"):
            content = env_file.getvalue().decode()
            for line in content.split("\n"):
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
                    
            has_aws_config = all([
                os.getenv("AWS_ACCESS_KEY_ID"),
                os.getenv("AWS_SECRET_ACCESS_KEY"),
                os.getenv("S3_BUCKET")
            ])
            
            if has_aws_config:
                st.success("AWS credentials loaded from .env file")
                st.info("Restart required to apply changes. Click 'Restart Application'")
                if st.button("Restart Application"):
                    st.rerun()
                return True
            else:
                st.error("Required AWS credentials not found in .env file")
                return False
                
    return has_aws_config

class PDFAnnotator:
    def __init__(self):
        """Initialize the PDF annotator using S3."""
        # No local data_dir for primary storage anymore
        self.s3_sync = S3DatasetSync() # Initialize S3 interface
        self.cache = {} # Cache for S3 list results, etc.
        self.temp_files = {} # Keep track of temporary PDF downloads {doc_id: temp_path}

        # Configure OpenAI client
        load_dotenv(override=True) # Load API keys
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

        if not self.s3_sync.is_enabled():
            st.error("S3 storage is not configured. Please set AWS credentials and S3_BUCKET environment variable.")

    def get_datasets(self):
        """Get list of available datasets directly from S3."""
        if not self.s3_sync.is_enabled():
            return ["default"] # Return default if S3 is off

        # Use cache if available
        if 'datasets' in self.cache:
            return self.cache['datasets']

        datasets = self.s3_sync.list_datasets()

        if not datasets:
            # Ensure 'default' dataset structure exists in S3 if none found
            self.s3_sync.ensure_dataset_structure("default")
            datasets = ["default"]

        self.cache['datasets'] = datasets # Cache the result
        return datasets

    def _ensure_dataset_exists(self, dataset_id):
        """Ensure the basic structure for a dataset exists in S3."""
        if self.s3_sync.is_enabled():
            self.s3_sync.ensure_dataset_structure(dataset_id)
            # Update dataset cache
            if 'datasets' in self.cache and dataset_id not in self.cache['datasets']:
                 self.cache['datasets'].append(dataset_id)
                 self.cache['datasets'].sort()
            elif 'datasets' not in self.cache:
                 self.cache['datasets'] = self.s3_sync.list_datasets() # Refresh cache

    def get_pdf_files(self, dataset_id, only_unannotated=False):
        """Get list of PDF files from S3 for the dataset."""
        if not self.s3_sync.is_enabled():
            return []

        # Use cache if available
        cache_key = f"{dataset_id}_pdfs"
        if cache_key in self.cache:
             pdf_files = self.cache[cache_key]
        else:
            pdf_files = self.s3_sync.list_pdfs_in_dataset(dataset_id)
            self.cache[cache_key] = pdf_files # Cache result

        if only_unannotated:
            # Filter out PDFs that already have annotations
            # This requires fetching annotations, which can be slow.
            # Consider optimizing this if performance is critical.
            annotations = self.get_annotations(dataset_id)
            annotated_doc_ids = {ann.get("doc_id") for ann in annotations if ann.get("doc_id")}
            unannotated = [
                pdf for pdf in pdf_files
                if os.path.splitext(pdf)[0] not in annotated_doc_ids
            ]
            return unannotated
        return pdf_files

    def _convert_image_to_pdf_bytes(self, image_bytes):
        """Convert image bytes to PDF bytes."""
        try:
            # Use BytesIO to handle image data in memory
            img_stream = BytesIO(image_bytes)
            img = Image.open(img_stream)

            # Convert to RGB if necessary (e.g., for RGBA PNGs)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Use img2pdf to convert in memory
            pdf_bytes = img2pdf.convert(img.filename) # img2pdf can take Image object
            return pdf_bytes
        except Exception as e:
            print(f"Error converting image to PDF bytes: {e}")
            # Fallback using PIL save to BytesIO
            try:
                pdf_stream = BytesIO()
                img.save(pdf_stream, "PDF", resolution=100.0)
                return pdf_stream.getvalue()
            except Exception as pil_e:
                print(f"Fallback image conversion failed: {pil_e}")
                return None

    def upload_pdf(self, uploaded_file, dataset_id):
        """Upload a PDF or image file (converted to PDF) directly to S3."""
        if uploaded_file is None or not self.s3_sync.is_enabled():
            return None

        file_uuid = str(uuid.uuid4())
        original_name = uploaded_file.name
        file_extension = os.path.splitext(original_name)[1].lower()
        new_filename = f"{file_uuid}.pdf" # Always save as PDF in S3
        s3_key = f"{self.s3_sync._get_pdf_prefix(dataset_id)}{new_filename}"

        file_bytes = uploaded_file.getvalue()
        upload_success = False

        if file_extension in ['.jpg', '.jpeg', '.png']:
            # Convert image to PDF bytes first
            pdf_bytes = self._convert_image_to_pdf_bytes(file_bytes)
            if pdf_bytes:
                upload_success = self.s3_sync.upload_bytes(pdf_bytes, s3_key, content_type='application/pdf')
            else:
                 return {"error": f"Failed to convert image {original_name} to PDF"}
        elif file_extension == '.pdf':
            # Upload PDF bytes directly
            upload_success = self.s3_sync.upload_bytes(file_bytes, s3_key, content_type='application/pdf')
        else:
            return {"error": f"Unsupported file type: {file_extension}"}

        if upload_success:
             # Invalidate PDF list cache for this dataset
             cache_key = f"{dataset_id}_pdfs"
             if cache_key in self.cache:
                 del self.cache[cache_key]

             return {
                 "original_name": original_name,
                 "saved_as": new_filename, # Filename in S3
                 "uuid": file_uuid,
                 "s3_key": s3_key
             }
        else:
            return {"error": f"Failed to upload {original_name} to S3"}

    def _get_temp_pdf_path(self, dataset_id, pdf_filename):
        """Gets path to a temporary local copy of the PDF, downloading if needed."""
        doc_id = os.path.splitext(pdf_filename)[0]
        if doc_id in self.temp_files and Path(self.temp_files[doc_id]).exists():
            return self.temp_files[doc_id] # Return cached path

        # Download from S3
        temp_path = self.s3_sync.download_pdf_to_temp(dataset_id, pdf_filename)
        if temp_path:
            self.temp_files[doc_id] = temp_path # Store path
            return temp_path
        else:
            st.error(f"Failed to download PDF '{pdf_filename}' from S3.")
            return None

    def display_pdf(self, dataset_id, pdf_filename, page_to_show=None):
        """Display PDF from S3 (via temporary local file) in Streamlit."""
        temp_pdf_path = self._get_temp_pdf_path(dataset_id, pdf_filename)
        if not temp_pdf_path:
            return 0 # Return 0 pages if download failed

        num_pages = 0
        try:
            with open(temp_pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                # Get number of pages using PyPDF2 from the temp file
                f.seek(0) # Reset file pointer
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)

            # Embed PDF (same logic as before, using base64 data)
            iframe_key = f"pdf_iframe_{dataset_id}_{pdf_filename}_{page_to_show}"
            pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}#page={page_to_show + 1 if page_to_show is not None else 1}"
                    width="100%" height="800" type="application/pdf" id="{iframe_key}"></iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)

        except FileNotFoundError:
             st.error(f"Temporary PDF file not found: {temp_pdf_path}")
             return 0
        except Exception as e:
            st.error(f"Error displaying PDF {pdf_filename}: {e}")
            return 0

        return num_pages

    def save_annotation(self, dataset_id, doc_id, question, answer, page_ids):
        """Save a new annotation by updating the JSONL file in S3."""
        if not self.s3_sync.is_enabled():
            st.error("S3 not configured. Cannot save annotation.")
            return None

        # Create the annotation object
        timestamp = int(time.time() * 1000)
        answer_page_idx = 0
        if page_ids:
            try:
                first_page_id = page_ids[0]
                page_suffix = first_page_id.split('_p')[-1]
                answer_page_idx = int(page_suffix)
            except (ValueError, IndexError):
                answer_page_idx = 0

        annotation = {
            "questionId": timestamp,
            "question": question,
            "doc_id": doc_id,
            "page_ids": page_ids,
            "answers": [answer],
            "answer_page_idx": answer_page_idx,
            "data_split": "test",
            "dataset_id": dataset_id
        }

        # --- Read-Modify-Write S3 Annotation File ---
        # 1. Download existing content
        existing_content = self.s3_sync.download_annotation_content(dataset_id)
        if existing_content is None: # Check for download error
             st.error("Failed to download existing annotations from S3. Cannot save.")
             return None

        # 2. Append new annotation
        new_line = json.dumps(annotation)
        updated_content = existing_content.strip() + '\n' + new_line + '\n'

        # 3. Upload updated content
        success = self.s3_sync.upload_annotation_content(dataset_id, updated_content)

        if success:
            # Invalidate annotation cache for this dataset
            cache_key = f"{dataset_id}_annotations"
            if cache_key in self.cache:
                del self.cache[cache_key]
            return annotation
        else:
            st.error("Failed to upload updated annotations to S3.")
            return None

    def get_annotations(self, dataset_id, doc_id=None):
        """Get annotations from the JSONL file in S3."""
        if not self.s3_sync.is_enabled():
            return []

        # Use cache if available and not filtering by doc_id
        cache_key = f"{dataset_id}_annotations"
        if doc_id is None and cache_key in self.cache:
            return self.cache[cache_key]

        content = self.s3_sync.download_annotation_content(dataset_id)
        if content is None: # Download error
            st.error(f"Failed to load annotations for dataset '{dataset_id}' from S3.")
            return []
        if not content.strip(): # Empty file
             return []

        annotations = []
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    annotation = json.loads(line)
                    # Filter by doc_id if provided
                    if doc_id is None or annotation.get("doc_id") == doc_id:
                        annotations.append(annotation)
                except json.JSONDecodeError:
                    st.warning(f"Error parsing annotation line {i+1} in dataset {dataset_id}.jsonl")
                    continue

        # Cache the full list if not filtered
        if doc_id is None:
            self.cache[cache_key] = annotations

        return annotations

    def get_dataset_stats(self, dataset_id):
        """Get statistics for a dataset from S3 data."""
        if not self.s3_sync.is_enabled():
            return {"total_pdfs": 0, "pdfs_with_qa": 0, "total_questions": 0}

        # Fetch data needed for stats
        pdf_files = self.get_pdf_files(dataset_id) # Uses cache if available
        annotations = self.get_annotations(dataset_id) # Uses cache if available

        # Calculate stats
        pdf_with_qa = {a.get("doc_id") for a in annotations if a.get("doc_id")}

        return {
            "total_pdfs": len(pdf_files),
            "pdfs_with_qa": len(pdf_with_qa),
            "total_questions": len(annotations)
        }

    def get_annotated_docs(self, dataset_id):
        """Get document IDs and filenames that have annotations in S3."""
        if not self.s3_sync.is_enabled():
            return []

        annotations = self.get_annotations(dataset_id) # Uses cache
        pdf_files = self.get_pdf_files(dataset_id) # Uses cache

        annotated_doc_ids = {a.get("doc_id") for a in annotations if a.get("doc_id")}

        annotated_docs = []
        for pdf_filename in pdf_files:
            doc_id = os.path.splitext(pdf_filename)[0]
            if doc_id in annotated_doc_ids:
                annotated_docs.append({
                    "doc_id": doc_id,
                    "filename": pdf_filename
                })

        return sorted(annotated_docs, key=lambda x: x["filename"])

    def extract_page_text(self, dataset_id, pdf_filename, page_number):
        """Extract text from a specific page of a PDF downloaded from S3."""
        temp_pdf_path = self._get_temp_pdf_path(dataset_id, pdf_filename)
        if not temp_pdf_path:
            return ""

        try:
            with open(temp_pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if 0 <= page_number < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_number]
                    return page.extract_text() or "" # Return empty string if extraction yields None
                else:
                    st.warning(f"Page number {page_number} out of range for {pdf_filename}")
                    return ""
        except Exception as e:
            st.error(f"Error extracting text from PDF {pdf_filename}: {str(e)}")
            return ""

    def extract_multiple_pages_text(self, dataset_id, pdf_filename, page_numbers):
        """Extract text from multiple pages of a PDF downloaded from S3."""
        temp_pdf_path = self._get_temp_pdf_path(dataset_id, pdf_filename)
        if not temp_pdf_path:
            return ""

        combined_text = ""
        try:
            with open(temp_pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_actual_pages = len(pdf_reader.pages)
                for page_number in sorted(list(set(page_numbers))): # Ensure unique and sorted
                    if 0 <= page_number < num_actual_pages:
                        page = pdf_reader.pages[page_number]
                        page_text = page.extract_text() or ""
                        combined_text += f"\n\n--- PAGE {page_number} ---\n\n{page_text}"
                    else:
                         st.warning(f"Page number {page_number} out of range for {pdf_filename}")
            return combined_text
        except Exception as e:
            st.error(f"Error extracting text from PDF {pdf_filename}: {str(e)}")
            return ""

    def auto_annotate(self, dataset_id, pdf_filename, page_numbers, num_questions=3):
        """Use LLM to generate Q&A pairs based on text from PDF pages (downloaded from S3)."""
        if isinstance(page_numbers, int):
            page_numbers = [page_numbers]

        if not page_numbers:
            return {"error": "No pages selected for annotation."}

        # --- Get text from relevant pages ---
        # Download PDF temporarily if not already cached locally
        temp_pdf_path = self._get_temp_pdf_path(dataset_id, pdf_filename)
        if not temp_pdf_path:
             return {"error": f"Could not retrieve PDF {pdf_filename} for auto-annotation."}

        # Sample pages if too many selected (same logic as before)
        MAX_PAGES_TO_PROCESS = 5
        sampled_pages = page_numbers
        if len(page_numbers) > MAX_PAGES_TO_PROCESS:
            # Simplified sampling: take first, last, and some in between
            step = len(page_numbers) // (MAX_PAGES_TO_PROCESS -1) if MAX_PAGES_TO_PROCESS > 1 else 1
            sampled_indices = [0] + [i * step for i in range(1, MAX_PAGES_TO_PROCESS - 1)] + [len(page_numbers) - 1]
            sampled_pages = sorted(list(set([page_numbers[i] for i in sampled_indices if 0 <= i < len(page_numbers)])))
            st.info(f"Processing a sample of {len(sampled_pages)} pages ({', '.join(map(str, sampled_pages))}) out of {len(page_numbers)} selected.")

        # Extract text from the sampled pages using the temporary file
        context_text = self.extract_multiple_pages_text(dataset_id, pdf_filename, sampled_pages)

        if not context_text.strip():
            return {"error": f"Could not extract text from the selected pages ({', '.join(map(str, sampled_pages))}) of {pdf_filename}."}

        # --- Call LLM (same logic as before) ---
        model = "mistralai/mixtral-8x7b-instruct"
        try:
            messages = []
            # System prompt (adjust as needed)
            messages.append({
                "role": "system",
                "content": "You are an expert assistant tasked with generating high-quality, specific question-answer pairs based *only* on the provided text content from PDF pages. Focus on details present in the text. Ensure questions require reading the text to answer and are not generic. Provide answers directly extracted or synthesized from the text. Identify the source page numbers for each answer. Format the output as a JSON array of objects, where each object has 'question', 'answer', and 'source_pages' (a list of integers) keys."
            })
            # User prompt
            messages.append({
                "role": "user",
                "content": f"Generate exactly {num_questions} question-answer pairs based on the following text extracted from pages {', '.join(map(str, sampled_pages))} of a PDF document. Ensure the 'source_pages' field accurately reflects the page numbers provided here ({', '.join(map(str, sampled_pages))}) from which the answer can be derived.\n\n--- TEXT START ---\n{context_text}\n--- TEXT END ---\n\nOutput:"
            })

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6, # Slightly lower temp for more factual Q&A
                # Add response_format for JSON output if supported by the model/API
                # response_format={"type": "json_object"}, # Uncomment if using compatible OpenAI API/model
            )

            content = response.choices[0].message.content

            # Extract JSON (same logic as before)
            try:
                qa_pairs = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
                if json_match:
                    try:
                        qa_pairs = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                         return {"error": "Could not parse JSON from LLM response markdown.", "raw_response": content}
                else:
                    # Attempt to find JSON array directly if not in markdown
                    json_match_direct = re.search(r'(\[.*?\])', content, re.DOTALL)
                    if json_match_direct:
                         try:
                             qa_pairs = json.loads(json_match_direct.group(1))
                         except json.JSONDecodeError:
                             return {"error": "Could not parse JSON from LLM response.", "raw_response": content}
                    else:
                         return {"error": "Could not find JSON array in LLM response.", "raw_response": content}

            # Validate structure and add original page numbers if missing
            validated_pairs = []
            for pair in qa_pairs:
                 if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                     # Ensure source_pages is a list of integers, default to original input if missing/invalid
                     source_pages = pair.get('source_pages')
                     if not isinstance(source_pages, list) or not all(isinstance(p, int) for p in source_pages):
                         # If LLM didn't provide valid pages, use the input sampled pages
                         pair['source_pages'] = sampled_pages
                     else:
                         # Filter pages to only those originally provided
                         pair['source_pages'] = [p for p in source_pages if p in page_numbers]

                     validated_pairs.append(pair)
                 else:
                     st.warning(f"Skipping invalid QA pair structure from LLM: {pair}")

            return {"qa_pairs": validated_pairs}

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"LLM Error: {e}")
            return {"error": f"Error generating Q&A pairs: {str(e)}", "traceback": error_details}

    def cleanup(self):
        """Clean up temporary files."""
        self.s3_sync.cleanup_temp_dir()
        self.temp_files = {}

def main():
    st.set_page_config(page_title="PDF Annotator (S3)", layout="wide")

    # Ensure temp dir exists (S3DatasetSync handles its own)
    # tempfile.tempdir = "annotator_temp" # Optional: set a specific root temp dir
    # Path(tempfile.gettempdir()).mkdir(exist_ok=True)

    # Check AWS configuration (remains the same)
    has_aws_config = setup_aws_config()

    # Initialize annotator (now uses S3)
    annotator = PDFAnnotator()

    # Register cleanup function to run on exit
    # NOTE: Streamlit doesn't have a perfect exit hook. This might not always run.
    # Consider manual cleanup button or periodic cleanup if needed.
    # import atexit
    # atexit.register(annotator.cleanup)

    st.title("PDF Question-Answer Annotator (S3 Storage)")

    if not annotator.s3_sync.is_enabled():
        st.sidebar.error("S3 Sync is not configured or enabled. Cannot proceed.")
        st.stop() # Stop execution if S3 isn't working

    # --- Sidebar ---
    st.sidebar.header("Dataset Selection")
    datasets = annotator.get_datasets()
    if not datasets:
         st.sidebar.warning("No datasets found in S3 or 'default' could not be created.")
         # Provide option to create default manually if needed
         if st.sidebar.button("Attempt to Create 'default' Dataset"):
             annotator._ensure_dataset_exists("default")
             st.rerun()
         selected_dataset = "default" # Fallback
    else:
        selected_dataset = st.sidebar.selectbox("Select Dataset", datasets, key="dataset_select")

    # Create new dataset
    with st.sidebar.expander("Create New Dataset"):
        new_dataset_name = st.text_input("New Dataset Name", key="new_dataset_name")
        if st.button("Create Dataset", key="create_dataset_btn"):
            if new_dataset_name and new_dataset_name not in datasets:
                annotator._ensure_dataset_exists(new_dataset_name)
                st.success(f"Created dataset structure for '{new_dataset_name}' in S3.")
                # Clear dataset cache and rerun
                if 'datasets' in annotator.cache: del annotator.cache['datasets']
                st.rerun()
            elif not new_dataset_name:
                 st.error("Please enter a dataset name.")
            else:
                st.error(f"Dataset '{new_dataset_name}' already exists.")

    # Mode selection
    st.sidebar.header("Mode")
    app_mode = st.sidebar.radio("Select Mode", ["Annotation Mode", "Dataset Viewer"], key="app_mode_radio")

    # Display dataset statistics
    with st.sidebar.expander("Dataset Statistics", expanded=True):
         stats = annotator.get_dataset_stats(selected_dataset)
         st.metric("Total PDFs", stats['total_pdfs'])
         st.metric("PDFs with Q/A", stats['pdfs_with_qa'])
         st.metric("Total Questions", stats['total_questions'])
         if st.button("Refresh Stats", key="refresh_stats_btn"):
             # Clear relevant caches
             if f"{selected_dataset}_pdfs" in annotator.cache: del annotator.cache[f"{selected_dataset}_pdfs"]
             if f"{selected_dataset}_annotations" in annotator.cache: del annotator.cache[f"{selected_dataset}_annotations"]
             st.rerun()

    # --- Annotation Mode ---
    if app_mode == "Annotation Mode":
        st.sidebar.header("Upload New PDF/Image")
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF or Image files (JPG/JPEG/PNG)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_files:
            if st.sidebar.button("Add Files to Dataset", key="add_files_btn"):
                progress_bar = st.sidebar.progress(0)
                success_count = 0
                error_count = 0
                with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
                    for i, file in enumerate(uploaded_files):
                        result = annotator.upload_pdf(file, selected_dataset)
                        if result and "error" not in result:
                            success_count += 1
                        elif result and "error" in result:
                            st.sidebar.error(f"Failed ({file.name}): {result['error']}")
                            error_count += 1
                        else:
                             st.sidebar.error(f"Failed ({file.name}): Unknown upload error.")
                             error_count += 1
                        progress_bar.progress((i + 1) / len(uploaded_files))

                if success_count > 0:
                    st.sidebar.success(f"Successfully uploaded {success_count} file(s).")
                if error_count == 0 and success_count > 0:
                     st.rerun() # Rerun only if all uploads were successful
                elif error_count > 0:
                     st.sidebar.warning(f"{error_count} file(s) failed to upload.")
                     # Optionally rerun even with errors if needed
                     # st.rerun()


        # PDF Filtering (Unannotated Only)
        st.sidebar.subheader("PDF Filter")
        # Initialize show_unannotated in session state if not present
        if 'show_unannotated' not in st.session_state:
            st.session_state.show_unannotated = False

        # Checkbox to toggle filter state
        show_unannotated = st.sidebar.checkbox("Show unannotated PDFs only",
                                          value=st.session_state.show_unannotated,
                                          key="unannotated_checkbox",
                                          on_change=lambda: setattr(st.session_state, 'pdf_index', 0)) # Reset index on filter change

        # Update session state based on checkbox
        st.session_state.show_unannotated = show_unannotated

        # Get PDF files based on filter
        pdf_files = annotator.get_pdf_files(selected_dataset, st.session_state.show_unannotated)

        if not pdf_files:
            st.sidebar.warning(f"No {'unannotated ' if st.session_state.show_unannotated else ''}PDF files found in dataset '{selected_dataset}'. Upload some files.")
            # Clear main area or show upload prompt
            st.info("Upload PDF or image files using the sidebar to begin annotation.")
            st.stop() # Stop further execution in this mode if no PDFs

        # PDF Navigation
        st.sidebar.subheader("PDF Navigation")
        if 'pdf_index' not in st.session_state or st.session_state.pdf_index >= len(pdf_files):
            st.session_state.pdf_index = 0 # Reset index if out of bounds

        col1, col2, col3 = st.sidebar.columns([1, 3, 1])
        with col1:
            if st.button("←", help="Previous PDF", key="prev_pdf_btn"):
                st.session_state.pdf_index = (st.session_state.pdf_index - 1 + len(pdf_files)) % len(pdf_files)
                # Reset page-specific state
                if 'current_page' in st.session_state: del st.session_state['current_page']
                if 'qa_suggestions' in st.session_state: del st.session_state['qa_suggestions'] # Clear suggestions for new PDF
                st.rerun()
        with col2:
            file_count_text = f"{st.session_state.pdf_index + 1}/{len(pdf_files)}"
            if st.session_state.show_unannotated: file_count_text += " (unannotated)"
            st.write(f"{file_count_text}:")
            st.caption(f"{pdf_files[st.session_state.pdf_index]}") # Use caption for smaller text
        with col3:
            if st.button("→", help="Next PDF", key="next_pdf_btn"):
                st.session_state.pdf_index = (st.session_state.pdf_index + 1) % len(pdf_files)
                 # Reset page-specific state
                if 'current_page' in st.session_state: del st.session_state['current_page']
                if 'qa_suggestions' in st.session_state: del st.session_state['qa_suggestions'] # Clear suggestions for new PDF
                st.rerun()

        selected_pdf = pdf_files[st.session_state.pdf_index]
        doc_id = os.path.splitext(selected_pdf)[0]

        # Display existing annotations for this document
        st.sidebar.subheader("Existing Annotations")
        existing_annotations = annotator.get_annotations(selected_dataset, doc_id) # Filter by doc_id
        if existing_annotations:
            for i, annotation in enumerate(existing_annotations):
                with st.sidebar.expander(f"Q{i+1}: {annotation['question'][:30]}..."):
                    st.markdown(f"**Q:** {annotation['question']}")
                    st.markdown(f"**A:** {annotation['answers'][0]}")
                    
                    # Add page numbers if available
                    page_info = ""
                    if 'page_ids' in annotation and annotation['page_ids']:
                        page_numbers = []
                        for page_id in annotation['page_ids']:
                            if '_p' in page_id:
                                try:
                                    page_numbers.append(int(page_id.split('_p')[1]))
                                except:
                                    pass
                        if page_numbers:
                            page_info = f"Pages: {', '.join(map(str, sorted(page_numbers)))}"
                            st.text(page_info)
        else:
            st.sidebar.info("No annotations for this document yet.")

        # --- Main Content Area ---
        col_main, col_annotate = st.columns([2, 1])

        with col_main:
            st.subheader(f"Document: {selected_pdf}")
            # Display PDF using the annotator method which handles temp download
            # Use session state to control the displayed page for navigation
            if 'current_page' not in st.session_state:
                 st.session_state.current_page = 0 # Default to first page

            num_pages = annotator.display_pdf(selected_dataset, selected_pdf, st.session_state.current_page)

            # Add page navigation controls below the PDF
            if num_pages > 1:
                 st.write("---")
                 nav_cols = st.columns([1,1,1,3,1,1,1])
                 with nav_cols[0]:
                     if st.button("⏮️ First", key="pg_first"):
                         st.session_state.current_page = 0
                         st.rerun()
                 with nav_cols[1]:
                     if st.button("⬅️ Prev", key="pg_prev"):
                         st.session_state.current_page = max(0, st.session_state.current_page - 1)
                         st.rerun()
                 with nav_cols[2]:
                     # Direct page input
                     target_page = st.number_input("Page", min_value=0, max_value=num_pages - 1, value=st.session_state.current_page, step=1, key="pg_num_input", label_visibility="collapsed")
                     if target_page != st.session_state.current_page:
                         st.session_state.current_page = target_page
                         st.rerun()

                 with nav_cols[3]:
                     st.write(f"Page **{st.session_state.current_page + 1}** of {num_pages}")

                 with nav_cols[4]:
                     if st.button("Next ➡️", key="pg_next"):
                         st.session_state.current_page = min(num_pages - 1, st.session_state.current_page + 1)
                         st.rerun()
                 with nav_cols[5]:
                     if st.button("Last ⏭️", key="pg_last"):
                         st.session_state.current_page = num_pages - 1
                         st.rerun()


        with col_annotate:
            st.subheader("Add Annotation")

            # Auto-Annotate Section
            with st.expander("Auto-Annotate with LLM", expanded=False):
                st.info("Generate Q&A pairs using an LLM based on selected page content.")
                if 'auto_annotate_pages' not in st.session_state:
                    st.session_state.auto_annotate_pages = []

                # Page selection for Auto-Annotate
                st.write("Select pages for LLM context:")
                num_cols = 5
                page_cols = st.columns(num_cols)
                selected_pages_for_llm = []

                # Select All / Clear All
                sel_cols = st.columns(2)
                with sel_cols[0]:
                    if st.button("Select All (Auto)", key="auto_select_all"):
                        st.session_state.auto_annotate_pages = list(range(num_pages))
                        st.rerun()
                with sel_cols[1]:
                    if st.button("Clear All (Auto)", key="auto_clear_all"):
                        st.session_state.auto_annotate_pages = []
                        st.rerun()

                # Page checkboxes
                for i in range(num_pages):
                    with page_cols[i % num_cols]:
                        is_selected = i in st.session_state.auto_annotate_pages
                        new_state = st.checkbox(f"{i}", value=is_selected, key=f"auto_page_cb_{i}")
                        if new_state:
                            selected_pages_for_llm.append(i)
                        elif i in st.session_state.auto_annotate_pages: # If unchecked
                             pass # Handled by rebuilding the list

                # Update state only if changed to avoid unnecessary reruns from checkboxes
                if set(selected_pages_for_llm) != set(st.session_state.auto_annotate_pages):
                     st.session_state.auto_annotate_pages = sorted(list(set(selected_pages_for_llm)))
                     # No rerun here, let button trigger action

                st.caption(f"Selected {len(st.session_state.auto_annotate_pages)} pages for LLM.")

                num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=10, value=3, key="num_llm_q")

                if st.button("Generate Q&A Pairs", key="generate_qa_btn"):
                    if not st.session_state.auto_annotate_pages:
                        st.warning("Please select at least one page for the LLM.")
                    else:
                        with st.spinner(f"Generating {num_questions} Q&A pairs from {len(st.session_state.auto_annotate_pages)} pages..."):
                            # Pass dataset_id and pdf_filename to auto_annotate
                            result = annotator.auto_annotate(
                                selected_dataset,
                                selected_pdf,
                                st.session_state.auto_annotate_pages,
                                num_questions
                            )
                            if "error" in result:
                                st.error(result["error"])
                                if "raw_response" in result:
                                    with st.expander("Raw LLM Response"): st.code(result["raw_response"])
                                if "traceback" in result:
                                     with st.expander("Traceback"): st.code(result["traceback"])
                            elif "qa_pairs" in result:
                                st.session_state.qa_suggestions = result["qa_pairs"]
                                st.success(f"Generated {len(result['qa_pairs'])} Q&A pairs!")
                                # Rerun to display suggestions
                                st.rerun()

            # Display LLM Suggestions
            if "qa_suggestions" in st.session_state and st.session_state.qa_suggestions:
                st.subheader("Review Generated Q&A")
                # Use index manipulation to handle removal during iteration
                indices_to_keep = list(range(len(st.session_state.qa_suggestions)))
                indices_to_remove = []

                for i in indices_to_keep:
                    qa_pair = st.session_state.qa_suggestions[i]
                    with st.expander(f"Suggestion {i+1}: {qa_pair['question'][:40]}...", expanded=True, key=f"suggest_exp_{i}"):
                        # Use unique keys for text areas within the loop
                        edited_question = st.text_area("Question", value=qa_pair['question'], key=f"q_suggest_{doc_id}_{i}")
                        edited_answer = st.text_area("Answer", value=qa_pair['answer'], key=f"a_suggest_{doc_id}_{i}")

                        # Page selection for this specific QA pair
                        st.write("Relevant pages for this pair:")
                        default_pages = qa_pair.get('source_pages', st.session_state.auto_annotate_pages) # Use LLM pages or original selection
                        # Use a unique session state key for each suggestion's pages
                        session_key_pages = f'qa_pages_{doc_id}_{i}'
                        if session_key_pages not in st.session_state:
                             st.session_state[session_key_pages] = default_pages

                        num_cols_qa = 5
                        qa_page_cols = st.columns(num_cols_qa)
                        selected_pages_for_qa = []

                        for p in range(num_pages):
                             with qa_page_cols[p % num_cols_qa]:
                                 is_selected_qa = p in st.session_state[session_key_pages]
                                 new_state_qa = st.checkbox(f"{p}", value=is_selected_qa, key=f"qa_page_cb_{doc_id}_{i}_{p}")
                                 if new_state_qa:
                                     selected_pages_for_qa.append(p)

                        # Update state if changed
                        if set(selected_pages_for_qa) != set(st.session_state[session_key_pages]):
                             st.session_state[session_key_pages] = sorted(list(set(selected_pages_for_qa)))
                             # No rerun needed here

                        st.caption(f"Selected {len(st.session_state[session_key_pages])} pages.")

                        # Add buttons to scroll to pages
                        if st.session_state[session_key_pages]:
                             page_buttons = st.columns(min(5, len(st.session_state[session_key_pages])))
                             for j, page in enumerate(sorted(st.session_state[session_key_pages])[:5]):
                                 with page_buttons[j]:
                                     if st.button(f"Go to {page}", key=f"goto_review_page_{doc_id}_{i}_{j}"):
                                         st.session_state.current_page = page
                                         st.rerun()

                        # Convert page numbers to page_ids
                        qa_page_ids = [f"{doc_id}_p{page}" for page in st.session_state[session_key_pages]]

                        if st.button(f"Add Suggestion {i+1} to Dataset", key=f"add_suggest_{doc_id}_{i}"):
                            if not edited_question or not edited_answer:
                                 st.error("Question and Answer cannot be empty.")
                            elif not qa_page_ids:
                                st.error("Please select at least one relevant page.")
                            else:
                                annotation = annotator.save_annotation(
                                    selected_dataset, doc_id, edited_question, edited_answer, qa_page_ids
                                )
                                if annotation:
                                    st.success(f"Suggestion {i+1} added!")
                                    indices_to_remove.append(i) # Mark for removal
                                    # Clear page selection state for this suggestion
                                    if session_key_pages in st.session_state: del st.session_state[session_key_pages]
                                    # Rerun needed to update UI and existing annotations list
                                    st.rerun()

                # Remove added suggestions from the list *after* iteration
                if indices_to_remove:
                    # Remove in reverse order to avoid index shifting issues
                    for index in sorted(indices_to_remove, reverse=True):
                        st.session_state.qa_suggestions.pop(index)
                    # If items were removed, a rerun might already be triggered by the button,
                    # but ensure one happens if the button didn't trigger it.
                    # st.rerun() # Usually handled by the button press rerun

            # Manual Annotation Form
            st.subheader("Manual Annotation")
            with st.form("manual_annotation_form", clear_on_submit=True):
                question = st.text_area("Question", height=100, key=f"manual_q_{doc_id}")
                answer = st.text_area("Answer", height=100, key=f"manual_a_{doc_id}")

                st.write("Select relevant pages:")
                # Use unique session state key for manual form pages
                manual_pages_key = f"manual_form_pages_{doc_id}"
                if manual_pages_key not in st.session_state:
                    st.session_state[manual_pages_key] = []

                num_cols_manual = 5
                manual_page_cols = st.columns(num_cols_manual)
                selected_pages_manual = []

                # Select All / Clear All for manual form
                sel_cols_man = st.columns(2)
                with sel_cols_man[0]:
                     if st.form_submit_button("Select All (Manual)", use_container_width=True):
                         st.session_state[manual_pages_key] = list(range(num_pages))
                         # Need to prevent full form submission here, just update state
                         # This is tricky with st.form. Consider moving buttons outside form.
                         # For now, user clicks this, then submits form.
                with sel_cols_man[1]:
                     if st.form_submit_button("Clear All (Manual)", use_container_width=True):
                         st.session_state[manual_pages_key] = []
                         # Similar issue as Select All

                # Page checkboxes for manual form
                for i in range(num_pages):
                    with manual_page_cols[i % num_cols_manual]:
                        is_selected_man = i in st.session_state[manual_pages_key]
                        # Checkbox inside form doesn't trigger immediate rerun
                        new_state_man = st.checkbox(f"{i}", value=is_selected_man, key=f"manual_page_cb_{doc_id}_{i}")
                        if new_state_man:
                            selected_pages_manual.append(i)

                # Update state on submit based on checkboxes checked *at submission time*
                # Note: Checkbox state might not be perfectly up-to-date if user clicks Select/Clear then Submit immediately.
                # This is a limitation of how st.form interacts with widgets.

                st.caption(f"Selected {len(st.session_state[manual_pages_key])} pages (update on save).") # Reflects state *before* submit

                submitted = st.form_submit_button("Save Manual Annotation")

                if submitted:
                    # Update selected pages based on checkbox state *now*
                    current_manual_selection = []
                    for i in range(num_pages):
                         if st.session_state[f"manual_page_cb_{doc_id}_{i}"]:
                             current_manual_selection.append(i)
                    st.session_state[manual_pages_key] = sorted(list(set(current_manual_selection)))


                    page_ids_manual = [f"{doc_id}_p{page}" for page in st.session_state[manual_pages_key]]

                    if not question or not answer:
                        st.error("Please fill in Question and Answer.")
                    elif not page_ids_manual:
                        st.error("Please select at least one relevant page.")
                    else:
                        annotation = annotator.save_annotation(selected_dataset, doc_id, question, answer, page_ids_manual)
                        if annotation:
                            st.success("Manual annotation saved successfully!")
                            # Clear page selection state for next manual entry
                            st.session_state[manual_pages_key] = []
                            # Rerun to update existing annotations list
                            st.rerun()
                        # else: save_annotation shows S3 error

    # --- Dataset Viewer Mode ---
    elif app_mode == "Dataset Viewer":
        st.header(f"Dataset Viewer: {selected_dataset}")

        # Get documents with annotations
        annotated_docs = annotator.get_annotated_docs(selected_dataset)

        if not annotated_docs:
            st.info(f"No annotated documents found in dataset '{selected_dataset}'. Annotate some documents in Annotation Mode.")
            st.stop()

        # Document Search (remains the same logic)
        st.sidebar.subheader("Document Search")
        search_query = st.sidebar.text_input("Search annotated documents", "", key="viewer_search")
        filtered_docs = annotated_docs
        if search_query:
            query = search_query.lower()
            filtered_docs = [
                doc for doc in annotated_docs
                if query in doc["filename"].lower() or query in doc["doc_id"].lower()
            ]
            if not filtered_docs:
                st.sidebar.warning(f"No documents found matching '{search_query}'")
                filtered_docs = annotated_docs # Show all if no match
            else:
                st.sidebar.info(f"Found {len(filtered_docs)} matching documents.")

        # Viewer Navigation
        st.sidebar.subheader("Document Navigation")
        if 'viewer_doc_index' not in st.session_state: st.session_state.viewer_doc_index = 0
        if st.session_state.viewer_doc_index >= len(filtered_docs): st.session_state.viewer_doc_index = 0 # Reset if out of bounds

        v_col1, v_col2, v_col3 = st.sidebar.columns([1, 3, 1])
        with v_col1:
            if st.button("← Prev Doc", key="viewer_prev_doc"):
                st.session_state.viewer_doc_index = (st.session_state.viewer_doc_index - 1 + len(filtered_docs)) % len(filtered_docs)
                st.session_state.viewer_page_to_show = None  # Reset page view
                st.rerun()
        with v_col2:
            current_doc = filtered_docs[st.session_state.viewer_doc_index]
            st.write(f"{st.session_state.viewer_doc_index + 1}/{len(filtered_docs)}:")
            st.caption(f"{current_doc['filename']}")
        with v_col3:
            if st.button("Next Doc →", key="viewer_next_doc"):
                st.session_state.viewer_doc_index = (st.session_state.viewer_doc_index + 1) % len(filtered_docs)
                st.session_state.viewer_page_to_show = None  # Reset page view
                st.rerun()

        # Display current document and its QA pairs
        doc = filtered_docs[st.session_state.viewer_doc_index]
        doc_id = doc["doc_id"]
        filename = doc["filename"]
        annotations = annotator.get_annotations(selected_dataset, doc_id) # Filter by doc_id

        # Layout for viewer
        v_col_pdf, v_col_qa = st.columns([2, 1])

        with v_col_pdf:
            st.subheader(f"Document: {filename}")
            # Use session state for page navigation in viewer
            if 'viewer_page_to_show' not in st.session_state:
                 st.session_state.viewer_page_to_show = 0 # Default to first page

            num_pages_viewer = annotator.display_pdf(selected_dataset, filename, st.session_state.viewer_page_to_show)

             # Add page navigation controls below the PDF for viewer
            if num_pages_viewer > 1:
                 st.write("---")
                 v_nav_cols = st.columns([1,1,1,3,1,1,1])
                 with v_nav_cols[0]:
                     if st.button("⏮️ First", key="v_pg_first"):
                         st.session_state.viewer_page_to_show = 0
                         st.rerun()
                 with v_nav_cols[1]:
                     if st.button("⬅️ Prev", key="v_pg_prev"):
                         st.session_state.viewer_page_to_show = max(0, st.session_state.viewer_page_to_show - 1)
                         st.rerun()
                 with v_nav_cols[2]:
                     v_target_page = st.number_input("Page", min_value=0, max_value=num_pages_viewer - 1, value=st.session_state.viewer_page_to_show, step=1, key="v_pg_num_input", label_visibility="collapsed")
                     if v_target_page != st.session_state.viewer_page_to_show:
                         st.session_state.viewer_page_to_show = v_target_page
                         st.rerun()
                 with v_nav_cols[3]:
                     st.write(f"Page **{st.session_state.viewer_page_to_show + 1}** of {num_pages_viewer}")
                 with v_nav_cols[4]:
                     if st.button("Next ➡️", key="v_pg_next"):
                         st.session_state.viewer_page_to_show = min(num_pages_viewer - 1, st.session_state.viewer_page_to_show + 1)
                         st.rerun()
                 with v_nav_cols[5]:
                     if st.button("Last ⏭️", key="v_pg_last"):
                         st.session_state.viewer_page_to_show = num_pages_viewer - 1
                         st.rerun()

        with v_col_qa:
            st.subheader(f"Annotations ({len(annotations)})")
            if not annotations:
                st.info("No annotations found for this document.")
            else:
                for i, annotation in enumerate(annotations):
                    with st.expander(f"Q{i+1}: {annotation['question'][:40]}...", expanded=(i==0)):
                        st.markdown(f"**Question:**")
                        st.markdown(f"> {annotation['question']}") # Use blockquote
                        st.markdown(f"**Answer:**")
                        st.markdown(f"> {annotation['answers'][0]}") # Use blockquote
                        st.markdown(f"**Pages:** {', '.join(annotation['page_ids'])}")

                        # Extract page numbers
                        page_numbers = []
                        for page_id in annotation['page_ids']:
                            try: page_numbers.append(int(page_id.split('_p')[1]))
                            except: pass

                        # Add button to jump to the first relevant page
                        if page_numbers:
                            first_page = min(page_numbers)
                            if st.button(f"Go to page {first_page}", key=f"goto_page_view_{i}"):
                                st.session_state.viewer_page_to_show = first_page
                                st.rerun()
                        st.markdown("---")

if __name__ == "__main__":
    main()
    # Consider adding cleanup here if atexit doesn't work reliably
    # PDFAnnotator().cleanup() # This creates a new instance, might not be ideal

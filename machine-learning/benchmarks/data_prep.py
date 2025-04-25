import os
import glob
from pydantic import BaseModel
import json
from PIL import Image
from typing import Dict, List, Tuple
from models import MPDocVQADataset, DocVQAItem
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


def group_pages_by_doc(dataset, images_dir):
    doc_pages = {}
    for item in dataset.data:
        doc_id = item.doc_id
        if doc_id not in doc_pages:
            doc_pages[doc_id] = []
            pattern = os.path.join(images_dir, f"{doc_id}_p*.jpg")
            pages = sorted(glob.glob(pattern), 
                           key=lambda x: int(x.split('_p')[1].split('.')[0]))
            doc_pages[doc_id] = pages
    return doc_pages

def find_image_files(doc_ids=None):
    image_dir = "images"
    if not os.path.exists(image_dir):
        print(f"Error: Images directory {image_dir} does not exist")
        return {}
    
    doc_images = {}
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    for img_path in tqdm(image_files, desc="Indexing images"):
        filename = os.path.basename(img_path)
        if "_p" in filename:
            doc_id = filename.split("_p")[0]
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            if doc_id not in doc_images:
                doc_images[doc_id] = []
            doc_images[doc_id].append(img_path)
    
    for doc_id in doc_images:
        doc_images[doc_id].sort(key=lambda x: int(os.path.basename(x).split("_p")[1].split(".")[0]))
    
    print(f"✓ Found {len(doc_images)} documents with {sum(len(images) for images in doc_images.values())} total images")
    return doc_images

def create_pdfs(doc_images, output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return {}
    
    pdf_paths = {}
    skipped = 0
    errors = 0
    
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    
    pdf_paths_lock = Lock()
    skipped_lock = Lock()
    errors_lock = Lock()
    
    def process_document(doc_item):
        doc_id, image_paths = doc_item
        local_pdf_path = None
        local_skipped = 0
        local_errors = 0
        
        if not image_paths:
            with skipped_lock:
                nonlocal skipped
                skipped += 1
            return None
            
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                local_errors += 1
                continue
        
        if images:
            pdf_path = os.path.join(output_dir, f"{doc_id}.pdf")
            try:
                images[0].save(pdf_path, "PDF", save_all=True, append_images=images[1:])
                if os.path.exists(pdf_path):
                    local_pdf_path = pdf_path
                else:
                    print(f"PDF was not created at {pdf_path}")
                    local_errors += 1
            except Exception as e:
                print(f"Error saving PDF for {doc_id}: {e}")
                local_errors += 1
        
        # Update shared counters
        with errors_lock:
            nonlocal errors
            errors += local_errors
        
        return (doc_id, local_pdf_path) if local_pdf_path else None
    
    # Use ThreadPoolExecutor to process documents in parallel
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_document, doc_images.items()),
            total=len(doc_images),
            desc="Creating PDFs"
        ))
    
    # Collect results
    for result in results:
        if result:
            doc_id, path = result
            pdf_paths[doc_id] = path
    
    print(f"✓ Created {len(pdf_paths)} PDFs (skipped: {skipped}, errors: {errors})")
    return pdf_paths

def filter_vague_questions(dataset, max_workers=20, log_queue=None):
    """
    Filter out vague visual questions from the dataset using an LLM.
    """
    total_count = len(dataset.data)
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("Warning: OPENROUTER_API_KEY not set, skipping question filtering")
        return dataset
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    
    def process_question(item):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": "You are a document question filtering assistant."},
                    {"role": "user", "content": f"Is this question vague and only asking about visual elements without specificity? Answer only YES or NO.\nQuestion: {item.question}\n\nExamples of vague questions: 'what is in this image', 'what xxxxx in/on 'this' page', 'what does the document show', `what is the table about`, 'what is on the bottom corner of the page', 'What is the first thing on the page?'.  Questions should be specific and at the document level NOT the page level. The questions must be about the document in general - and not be ambiguous or vague."}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return (item, "YES" not in result)
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return (item, True)
    
    print(f"Filtering {len(dataset.data)} questions in parallel...")
    
    # Process questions with progress bar
    results = []
    with tqdm(total=len(dataset.data), desc="Filtering questions") as pbar:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(dataset.data))) as executor:
            futures = []
            for item in dataset.data:
                future = executor.submit(process_question, item)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)
            
            for future in futures:
                results.append(future.result())
    
    filtered_data = [item for item, keep in results if keep]
    filtered_count = len(dataset.data) - len(filtered_data)
    
    dataset.data = filtered_data
    print(f"Filtered out {filtered_count} vague questions out of {total_count} total questions")
    
    return dataset

def load_jsonl_dataset(dataset_path):
    """Load a dataset from a JSONL file."""
    print(f"Loading dataset from {dataset_path}...")
    data_items = []
    
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Make sure we have all required fields
                        if 'data_split' not in item and 'dataset_id' in item:
                            item['data_split'] = item['dataset_id']
                        if 'ground_truths' in item and 'answers' not in item:
                            item['answers'] = item['ground_truths']
                        data_items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line: {e}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        # Extract dataset name from path to suggest the right structure
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_path)))
        print(f"Make sure your data is organized as data/{dataset_name}/vals/{dataset_name}.jsonl")
        raise
    
    # Create a dataset object with the loaded items
    dataset_obj = MPDocVQADataset(
        dataset_name="document_qa",
        dataset_version=1.0,
        dataset_split="test",
        data=[DocVQAItem(**item) for item in data_items]
    )
    print(f"✓ Loaded dataset with {len(dataset_obj.data)} questions")
    return dataset_obj

def prepare_data(dataset_path, max_docs=None, filter_questions=True, max_workers=20, log_queue=None, dataset_name=None):
    start_time = time.time()
    print(f"Loading dataset from {dataset_path}...")
    
    # Check if the file is JSONL or JSON based on extension
    if dataset_path.endswith('.jsonl'):
        dataset_obj = load_jsonl_dataset(dataset_path)
    else:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        dataset_obj = MPDocVQADataset(**dataset)
    
    # Use provided dataset_name or extract it from the path
    if not dataset_name:
        # Try to extract dataset name from path
        # For path like data/my_dataset/vals/my_dataset.jsonl
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_path)))
        # If that doesn't work (e.g., for old paths), use the dataset's name
        if not dataset_name or dataset_name == 'data':
            dataset_name = dataset_obj.dataset_name
    
    print(f"Using dataset name: {dataset_name}")
    
    unique_doc_ids = set()
    for item in dataset_obj.data:
        unique_doc_ids.add(item.doc_id)
    
    if max_docs is not None:
        limited_doc_ids = list(unique_doc_ids)[:max_docs]
        dataset_obj.data = [item for item in dataset_obj.data if item.doc_id in limited_doc_ids]
        print(f"✓ Loaded dataset with {len(dataset_obj.data)} questions (limited to {max_docs} unique docs)")
    else:
        print(f"✓ Loaded dataset with {len(dataset_obj.data)} questions")
    
    if filter_questions:
        dataset_obj = filter_vague_questions(dataset_obj, max_workers, log_queue)
    
    # Extract page IDs for each document
    doc_pages = {}
    for item in tqdm(dataset_obj.data, desc="Extracting pages"):
        if item.doc_id not in doc_pages:
            doc_pages[item.doc_id] = []
        for page_id in item.page_ids:
            if page_id not in doc_pages[item.doc_id]:
                doc_pages[item.doc_id].append(page_id)
    
    print(f"✓ Found {len(doc_pages)} unique documents referenced in dataset")
    
    # Find PDF paths in the correct directory structure: data/dataset_name/pdfs
    pdf_paths = {}
    pdf_dir = os.path.join("data", dataset_name, "pdfs")
    vals_dir = os.path.join("data", dataset_name, "vals")
    
    # Ensure these directories exist
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(vals_dir, exist_ok=True)
    
    print(f"Looking for PDFs in: {pdf_dir}")
    
    # Check multiple possible locations
    pdf_locations = [
        pdf_dir,  # Standard location
        os.path.join("data", dataset_name),  # Simplified location
        "pdfs",   # Root pdfs directory
        "data/pdfs",  # Alternative location
    ]

    for doc_id in doc_pages:
        found = False
        for location in pdf_locations:
            pdf_path = os.path.join(location, f"{doc_id}.pdf")
            if os.path.exists(pdf_path):
                pdf_paths[doc_id] = pdf_path
                found = True
                break
        
        if not found:
            print(f"⚠️ Warning: Could not find PDF for document {doc_id}")

    if len(pdf_paths) < len(doc_pages):
        print(f"⚠️ Warning: Only found PDFs for {len(pdf_paths)}/{len(doc_pages)} documents")
        # If we have no PDFs, try creating them from images
        if len(pdf_paths) == 0 and os.path.exists("images"):
            print("Attempting to create PDFs from images...")
            doc_images = find_image_files(doc_ids=list(doc_pages.keys()))
            if doc_images:
                pdf_paths = create_pdfs(doc_images, pdf_dir)
    
    elapsed = time.time() - start_time
    print(f"Data preparation completed in {elapsed:.2f} seconds")
    
    return dataset_obj, doc_pages, pdf_paths

def load_dataset(dataset_path):
    """Load a dataset from a JSON file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    dataset_obj = MPDocVQADataset(**dataset)
    print(f"✓ Loaded dataset with {len(dataset_obj.data)} questions")
    return dataset_obj

def prefilter_questions(input_path, output_path, max_workers=20):
    """
    Pre-filter vague questions and save the filtered dataset to a new file.
    
    Args:
        input_path: Path to the input dataset JSON file
        output_path: Path to save the filtered dataset
        max_workers: Maximum number of parallel workers for filtering
    """
    start_time = time.time()
    print(f"Pre-filtering questions from {input_path}...")
    
    # Load the dataset
    dataset_obj = load_dataset(input_path)
    original_count = len(dataset_obj.data)
    
    # Filter vague questions
    filtered_dataset = filter_vague_questions(dataset_obj, max_workers)
    filtered_count = original_count - len(filtered_dataset.data)
    
    # Save the filtered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered_dataset.dict(), f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"Pre-filtering completed in {elapsed:.2f} seconds")
    print(f"Filtered out {filtered_count} questions ({filtered_count/original_count:.1%})")
    print(f"Saved filtered dataset with {len(filtered_dataset.data)} questions to {output_path}")
    
    return filtered_dataset

if __name__ == "__main__":
    load_dotenv(override=True)
    # Create a pre-filtered evaluation dataset
    prefilter_questions("qas/val.json", "qas/eval.json")


import os
import json
import time
import argparse
import shutil
from processors import (
    ChunkrProcessorHTML, ChunkrProcessorMarkdown, DirectImageProcessor,
    ChunkrProcessorFlash, ChunkrMultiModalProcessor, ChunkrHTMLOcrAutoProcessor,
    LLMProcessorMD, MODELS
)
from utils import log_message, connect_redis
from openai import OpenAI
from dotenv import load_dotenv

EXTRACTOR_KEYS = [
    "chunkr_html",    
    "chunkr_multi_modal",
    # "chunkr_html_ocr_auto",
    # "chunkr_markdown",
    "flash",
    "direct",
    "gemini_pro_2.5",
    "llama-4-maverick",
    # "flash_lite",
    # "flash_md",
    # "gemini_pro",
    # "flash_thinking",
    # "sonnet_thinking",
    # "sonnet37",
    # "sonnet35",
    # "chunkr_flash",
    # Add new processor keys
    # "mistral24",
    # "gemma3",
    # "phi4",
    # "qwen72b",
    # "qwen7b",
    # "gpt4o",
    # "llama390b"
]


def initialize_processors():
    load_dotenv(override=True)
    """Initialize all document processors"""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENROUTER_API_KEY")
    chunkr_api_key = os.environ.get("CHUNKR_API_KEY")
    
    if not openai_api_key:
        log_message("No OpenAI API key found. Please set OPENAI_API_KEY environment variable.", status="error")
        return None
        
    if not chunkr_api_key:
        log_message("No Chunkr API key found. Please set CHUNKR_API_KEY environment variable.", status="error")
        return None
    
    openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openai_api_key)
    
    # Initialize processors
    processors = {}
    
    # Map extractor keys to model keys for LLM processors
    model_mapping = {
        "flash": "gemini_flash",
        "flash_lite": "gemini_flash_lite",
        "gemini_pro": "gemini_pro",
        # "flash_thinking": "gemini_flash_thinking",
        "sonnet_thinking": "claude_sonnet_thinking",
        "sonnet37": "claude_sonnet_37",
        "sonnet35": "claude_sonnet_35",
        "mistral24": "mistral_24",
        "gemma3": "gemma_3_27b_instruct",
        "phi4": "Phi_4",
        # "qwen72b": "qwen_2.5_72B",
        "qwen7b": "qwen_2.5_7B",
        "gpt4o": "openai_gpt_4o",
        "llama390b": "meta_llama3_90B"
    }
    
    # Initialize Chunkr processors
    chunkr_processors = {
        "chunkr_html": ChunkrProcessorHTML(chunkr_api_key),
        "chunkr_markdown": ChunkrProcessorMarkdown(chunkr_api_key),
        "chunkr_flash": ChunkrProcessorFlash(chunkr_api_key),
        "chunkr_multi_modal": ChunkrMultiModalProcessor(chunkr_api_key),
        "chunkr_html_ocr_auto": ChunkrHTMLOcrAutoProcessor(chunkr_api_key),
        "direct": DirectImageProcessor()
    }
    
    # Add all enabled processors to the processors dictionary
    for key in EXTRACTOR_KEYS:
        if key in chunkr_processors:
            processors[key] = chunkr_processors[key]
            log_message(f"Initialized Chunkr processor: {key}")
        elif key in model_mapping:
            model_key = model_mapping[key]
            processors[key] = LLMProcessorMD(openai_client, model_key)
            log_message(f"Initialized LLM processor: {key} using model {model_key}")
    
    return processors

def process_document(doc_id, pdf_path, output_dir, processors, question_id=None):
    """Process a document through all extraction processors"""
    results = {}
    
    # Get Redis connection for adding to scoring queue
    r = connect_redis()
    
    # Get run_id from output_dir path
    path_parts = output_dir.split(os.sep)
    run_id = path_parts[-2] if len(path_parts) >= 2 else None
    
    # Check for global cache setting
    use_global_cache = r.get(f"benchmark:{run_id}:use_global_cache") == "true"
    global_cache_dir = r.get(f"benchmark:{run_id}:global_cache_dir")
    
    log_message(f"Processing document {doc_id} with {len(processors)} processors")
    
    # Process each processor sequentially
    for processor_name, processor in processors.items():
        try:
            log_message(f"Running {processor_name} on {doc_id}")
            
            # Define global cache path
            global_cache_path = None
            if use_global_cache and global_cache_dir:
                global_cache_path = os.path.join(global_cache_dir, f"{doc_id}_{processor_name}_result.json")
            
            # Check file cache first - local cache
            cache_path = os.path.join(output_dir, f"{doc_id}_{processor_name}_result.json")
            
            # First check global cache
            result = None
            cache_used = False
            
            if use_global_cache and global_cache_path and os.path.exists(global_cache_path):
                try:
                    with open(global_cache_path, 'r') as f:
                        cached_data = json.load(f)
                        result = cached_data.get("result", cached_data)
                    log_message(f"Using global cached {processor_name} result for {doc_id}")
                    
                    # Copy to local cache
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    shutil.copy(global_cache_path, cache_path)
                    
                    cache_used = True
                except Exception as e:
                    log_message(f"Error reading global cache {global_cache_path}: {str(e)}", status="warning")
            
            # If global cache not available, check local cache
            if not cache_used and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                        result = cached_data.get("result", cached_data)
                    log_message(f"Using local cached {processor_name} result for {doc_id}")
                    
                    # Copy to global cache if enabled
                    if use_global_cache and global_cache_path:
                        os.makedirs(os.path.dirname(global_cache_path), exist_ok=True)
                        shutil.copy(cache_path, global_cache_path)
                    
                    cache_used = True
                except Exception as e:
                    log_message(f"Error reading local cache {cache_path}: {str(e)}", status="warning")
            
            # If no cache hit, process the document
            if not cache_used:
                # Process with the processor
                result = processor.process_document(pdf_path, output_dir)
                
                # Only cache and add to scoring queue if result is not None
                if result is not None:
                    # Cache to local file
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'w') as f:
                        json.dump({"result": result}, f)
                    
                    # Cache to global file if enabled
                    if use_global_cache and global_cache_path:
                        os.makedirs(os.path.dirname(global_cache_path), exist_ok=True)
                        with open(global_cache_path, 'w') as f:
                            json.dump({"result": result}, f)
                    
                    log_message(f"Saved {processor_name} result for {doc_id} to cache")
                else:
                    log_message(f"Processor {processor_name} failed to extract content from {doc_id}", status="warning")
            
            # Store this result in our results dictionary and add to scoring queue ONLY if result is not None
            if result is not None:
                results[processor_name] = result
                
                # Immediately add to scoring queue for this processor
                if run_id:
                    scoring_task = {
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "processor_name": processor_name,
                        "timestamp": time.time()
                    }
                    
                    # Add question_id to the scoring task if provided
                    if question_id:
                        scoring_task["question_id"] = question_id
                        
                    r.rpush("benchmark:scoring_queue", json.dumps(scoring_task))
                    log_message(f"Added {processor_name} result for {doc_id} to scoring queue")
            else:
                log_message(f"Skipping scoring for {processor_name} on {doc_id} due to extraction failure", status="warning")
        
        except Exception as e:
            log_message(f"Error processing {processor_name} for {doc_id}: {str(e)}", status="error")
            import traceback
            log_message(traceback.format_exc(), status="error")
        
        # Force garbage collection after each processor
        import gc
        gc.collect()
    
    # Store full results in Redis for reference
    if run_id:
        r.set(f"benchmark:{run_id}:extraction:{doc_id}", json.dumps(results))
    
    return results

def test_extractor():
    """Test function to process a single document from the extraction queue"""
    # Get Redis connection
    r = connect_redis()
    
    # Get a document from the queue
    task_data = r.lpop("benchmark:extraction_queue")
    if not task_data:
        log_message("No documents in queue for testing", status="error")
        return
    
    # Put the task back at the front of the queue
    r.lpush("benchmark:extraction_queue", task_data)
    
    # Process just one document by modifying the extractor loop
    process_single_doc = True
    extractor_loop(process_single_doc=True)
    
    log_message("Test extraction completed for one document")

def extractor_loop(process_single_doc=False):
    """Main loop for the document extractor"""
    # Initialize processors
    processors = initialize_processors()
    if not processors:
        log_message("Failed to initialize processors", status="error")
        return
    
    r = connect_redis()
    
    log_message("Starting document extractor loop...")
    
    while True:
        try:
            # Get a document from the queue
            task_data = r.lpop("benchmark:extraction_queue")
            if not task_data:
                log_message("No documents in queue, waiting...")
                time.sleep(5)
                continue
            
            # Parse task data
            task = json.loads(task_data)
            doc_id = task["doc_id"]
            run_id = task["run_id"]
            question_id = task.get("question_id")  # Get question_id if it exists
            
            log_message(f"Processing document {doc_id} for run {run_id}", run_id=run_id)
            if question_id:
                log_message(f"Processing specific question {question_id}", run_id=run_id)
            
            # Get config for this run
            config_data = r.get(f"benchmark:{run_id}:config")
            if not config_data:
                log_message(f"Config not found for run {run_id}", status="error", run_id=run_id)
                continue
            
            config = json.loads(config_data)
            processor_results_dir = config.get("processor_results_dir")
            
            # Create document output directory
            doc_output_dir = processor_results_dir
            os.makedirs(doc_output_dir, exist_ok=True)
            
            # Get document path
            dataset_info = r.get(f"benchmark:{run_id}:dataset_info")
            if not dataset_info:
                log_message(f"Dataset info not found for run {run_id}", status="error", run_id=run_id)
                continue
            
            pdf_paths = json.loads(dataset_info).get("pdf_paths", {})
            pdf_path = pdf_paths.get(doc_id)
            
            if not pdf_path:
                log_message(f"PDF path not found for document {doc_id}", status="error", run_id=run_id)
                # Continue with next doc in queue rather than stopping completely
                r.incr(f"benchmark:{run_id}:completed_extractions")
                continue
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                log_message(f"PDF file does not exist at path: {pdf_path}", status="error", run_id=run_id)
                r.incr(f"benchmark:{run_id}:completed_extractions")
                continue
                
            log_message(f"Processing PDF {pdf_path} for document {doc_id}", run_id=run_id)
            process_document(doc_id, pdf_path, doc_output_dir, processors, question_id)
            
            # Increment completed extractions counter
            r.incr(f"benchmark:{run_id}:completed_extractions")
            
            log_message(f"Document {doc_id} extraction completed, added to scoring queue", run_id=run_id)
            
            # If we're only processing a single document for testing, break after one
            if process_single_doc:
                log_message("Test mode: processed one document, exiting loop")
                break
            
        except KeyboardInterrupt:
            log_message("Extractor interrupted by user")
            break
            
        except Exception as e:
            log_message(f"Unexpected error in extractor loop: {str(e)}", status="error")
            import traceback
            log_message(traceback.format_exc(), status="error")
            time.sleep(5)  # Avoid rapid error loops
            
            # If we're in test mode, don't continue after an error
            if process_single_doc:
                break

def main():
    parser = argparse.ArgumentParser(description='Document Extractor')
    parser.add_argument('--test', action='store_true', help='Run a single test extraction')
    args = parser.parse_args()
    
    if args.test:
        test_extractor()
    else:
        # Start the extractor loop
        extractor_loop()

if __name__ == "__main__":
    main()
import os
import sys
import time
import argparse
import json
import redis
import uuid
import glob
from data_prep import prepare_data
from utils import log_message, connect_redis
from extractor import EXTRACTOR_KEYS
import shutil

def setup_benchmark(args):
    """Set up benchmark in Redis and return run_id"""
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    
    # Create Redis connection
    r = connect_redis()
    
    # Create run directory (remove any "benchmark" prefix since we're already in the benchmark directory)
    data_dir = os.path.join("runs", run_id)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for various outputs
    processor_results_dir = os.path.join(data_dir, "processor_results")
    scoring_results_dir = os.path.join(data_dir, "scoring_results")
    logs_dir = os.path.join(data_dir, "logs")
    
    os.makedirs(processor_results_dir, exist_ok=True)
    os.makedirs(scoring_results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure global cache
    global_cache_dir = os.path.join("data", "global_cache")
    os.makedirs(global_cache_dir, exist_ok=True)
    
    # Save cache info to Redis
    r.set(f"benchmark:{run_id}:global_cache_dir", global_cache_dir)
    
    # Let coordinator know to check cache
    r.set(f"benchmark:{run_id}:use_global_cache", "true")
    
    # Store configuration in Redis
    config = {
        "timestamp": time.time(),
        "dataset_name": args.dataset_name,
        "data_dir": data_dir,
        "processor_results_dir": processor_results_dir,
        "scoring_results_dir": scoring_results_dir,
        "logs_dir": logs_dir,
        "dataset_path": args.dataset_path
    }
    
    r.set(f"benchmark:{run_id}:config", json.dumps(config))
    
    # Prepare data - find the correct dataset file
    log_message(f"Loading dataset...")
    
    # Fix: Construct proper dataset path when args.dataset_path is a directory
    dataset_path = args.dataset_path
    print(f"Dataset path: {dataset_path}")
    # If dataset_path is a directory, construct the path to the JSONL file
    if os.path.isdir(dataset_path):
        # Construct path following the pattern: datasetpath/datasetname/vals/datasetname.jsonl
        dataset_path = os.path.join(dataset_path, args.dataset_name, "vals", f"{args.dataset_name}.jsonl")
        log_message(f"Using dataset file: {dataset_path}")
    
    # Verify the dataset file exists
    if not os.path.exists(dataset_path):
        log_message(f"Dataset file not found at {dataset_path}", status="error")
        sys.exit(1)
    # You may need to adjust prepare_data to handle JSONL files if it only handles JSON
    dataset, doc_pages, pdf_paths = prepare_data(
        dataset_path, 
        max_docs=args.max_docs,
        dataset_name=args.dataset_name,
        filter_questions=False
    )
    
    doc_ids = sorted(list(set(item.doc_id for item in dataset.data)))
    if args.max_docs:
        doc_ids = doc_ids[:args.max_docs]
    
    # Save dataset info to file and Redis
    dataset_info = {
        "doc_ids": doc_ids,
        "doc_pages": {k: v for k, v in doc_pages.items() if k in doc_ids},
        "pdf_paths": {k: v for k, v in pdf_paths.items() if k in doc_ids}
    }
    
    with open(os.path.join(data_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)
    
    r.set(f"benchmark:{run_id}:dataset_info", json.dumps(dataset_info))
    
    # Create questions by document dictionary
    questions_by_doc = {}
    for item in dataset.data:
        doc_id = item.doc_id
        if doc_id in doc_ids:
            if doc_id not in questions_by_doc:
                questions_by_doc[doc_id] = []
            # Include the question ID in the question data
            questions_by_doc[doc_id].append({
                "question": item.question,
                "question_id": item.questionId,  # Add question ID here
                "answers": item.answers
            })
    
    # Save questions by document to Redis
    r.set(f"benchmark:{run_id}:questions_by_doc", json.dumps(questions_by_doc))
    
    # Add all documents to the extraction queue
    for doc_id in doc_ids:
        # Add to extraction queue with all question IDs for this document
        task = {
            "run_id": run_id,
            "doc_id": doc_id,
            "question_ids": [q["question_id"] for q in questions_by_doc.get(doc_id, [])],  # Include all question IDs
            "timestamp": time.time()
        }
        r.rpush("benchmark:extraction_queue", json.dumps(task))
    
    # Set both keys for compatibility
    r.set(f"benchmark:{run_id}:total_docs", len(doc_ids))
    r.set(f"benchmark:{run_id}:total_tasks", len(doc_ids))
    r.set(f"benchmark:{run_id}:completed_extractions", 0)
    r.set(f"benchmark:{run_id}:completed_scoring", 0)
    
    log_message(f"Added {len(doc_ids)} tasks to extraction queue for run {run_id}")
    return run_id

def monitor_progress(run_id):
    """Monitor the progress of the benchmark"""
    r = connect_redis()
    
    # Get total documents
    total_docs = int(r.get(f"benchmark:{run_id}:total_docs") or 0)
    
    # Check if there are documents to process
    if total_docs == 0:
        log_message(f"No documents to process for run {run_id}. Check your dataset.", status="warning", run_id=run_id)
        return
    
    # Get config to determine which processors are being used
    config = json.loads(r.get(f"benchmark:{run_id}:config"))
    data_dir = config["data_dir"]
    
    # Get list of processors from Redis or use a default list
    processors_list = json.loads(r.get(f"benchmark:{run_id}:processors") or "[]")
    if not processors_list:
        # Fallback to default processor list if not found in Redis
        processors_list = EXTRACTOR_KEYS
        # Store for future reference
        r.set(f"benchmark:{run_id}:processors", json.dumps(processors_list))
    
    num_processors = len(processors_list)
    log_message(f"Monitoring progress with {num_processors} processors", run_id=run_id)
    
    # Monitor progress
    log_message(f"Monitoring progress of {total_docs} documents", run_id=run_id)
    
    # Get total questions count
    questions_by_doc = json.loads(r.get(f"benchmark:{run_id}:questions_by_doc"))
    total_questions = sum(len(questions) for questions in questions_by_doc.values())
    log_message(f"Total questions: {total_questions}", run_id=run_id)
    
    # Calculate total expected extraction tasks (processors × documents)
    total_expected_extractions = total_docs * num_processors
    log_message(f"Expected extraction tasks: {total_expected_extractions}", run_id=run_id)
    
    # Calculate total expected scoring tasks (processors × questions)
    total_expected_scorings = total_questions * num_processors
    log_message(f"Expected scoring tasks: {total_expected_scorings} ({total_questions} questions × {num_processors} processors)", run_id=run_id)
    
    # Store this for reference
    r.set(f"benchmark:{run_id}:total_expected_extractions", total_expected_extractions)
    r.set(f"benchmark:{run_id}:total_expected_scorings", total_expected_scorings)
    
    # Track documents that need to be reprocessed
    reprocess_docs = set()
    retry_counts = {}
    max_retries = 3
    
    # Path to results file
    # results_jsonl_path = os.path.join(data_dir, "benchmark_results.jsonl")
    
    # Set a maximum runtime to prevent infinite loops (4 hours)
    max_runtime = 4 * 60 * 60  # 4 hours in seconds
    start_time = time.time()
    
    while True:
        # Check if we've exceeded the maximum runtime
        if time.time() - start_time > max_runtime:
            log_message(f"Maximum runtime exceeded. Collecting results and exiting.", run_id=run_id)
            break
            
        # Count completed scoring tasks by checking the results file
        completed_scorings = 0
        processed_docs = set()
        processed_questions = set()
        
        # if os.path.exists(results_jsonl_path):
        #     try:
        #         with open(results_jsonl_path, 'r') as f:
        #             for line in f:
        #                 try:
        #                     result = json.loads(line)
        #                     completed_scorings += 1
        #                     processed_docs.add(result.get("doc_id"))
        #                     processed_questions.add((result.get("doc_id"), result.get("question_id")))
        #                 except json.JSONDecodeError:
        #                     continue
        #     except Exception as e:
        #         log_message(f"Error reading results file: {str(e)}", status="error", run_id=run_id)
        
        # Check extraction queue length
        extraction_queue_length = r.llen("benchmark:extraction_queue")
        scoring_queue_length = r.llen("benchmark:scoring_queue")
        
        # Log progress
        log_message(f"Progress: {len(processed_docs)}/{total_docs} documents, {completed_scorings}/{total_expected_scorings} scoring tasks", run_id=run_id)
        log_message(f"Queue status: {extraction_queue_length} extraction tasks, {scoring_queue_length} scoring tasks", run_id=run_id)
        
        # Save interim results every minute
        save_interim_results(run_id, data_dir)
        
        # Check if benchmark is complete - all documents processed and scored
        if completed_scorings >= total_expected_scorings:
            log_message(f"Benchmark complete! All {num_processors} processors have processed all {total_docs} documents with {total_questions} questions.", run_id=run_id)
            break
        

        
        # Wait for 10 seconds before checking again
        time.sleep(10)
    
    # Collect final results
    collect_results(run_id)
    
    log_message(f"Benchmark run {run_id} completed", run_id=run_id)
    log_message(f"Dashboard is still running. Keep this process alive to view results in the dashboard.", run_id=run_id)
    
    # Instead of exiting, go into a waiting state to keep the process alive
    # This allows the dashboard to continue running
    r.set(f"benchmark:{run_id}:status", "completed")
    log_message(f"Coordinator entering wait mode. Press Ctrl+C to exit when done viewing results.", run_id=run_id)
    
    try:
        while True:
            time.sleep(3600)  # Sleep for an hour at a time
    except KeyboardInterrupt:
        log_message(f"Coordinator exiting due to user request", run_id=run_id)

def save_interim_results(run_id, data_dir):
    """Save interim results to file"""
    r = connect_redis()
    
    # Save any extraction results that exist
    extraction_keys = r.keys(f"benchmark:{run_id}:extraction:*")
    # for key in extraction_keys:
    #     doc_id = key.split(":")[-1]
    #     extraction_data = r.get(key)
        
    #     if extraction_data:
    #         doc_output_dir = os.path.join(data_dir, "processor_results", doc_id)
    #         os.makedirs(doc_output_dir, exist_ok=True)
            
    #         result_file = os.path.join(doc_output_dir, "extraction_results.json")
    #         try:
    #             with open(result_file, 'w') as f:
    #                 json.dump(json.loads(extraction_data), f)
    #         except Exception as e:
    #             log_message(f"Error saving interim extraction results: {e}", status="error", run_id=run_id)
    
    # Save any scoring results that exist
    result_keys = r.keys(f"benchmark:{run_id}:result:*")
    for key in result_keys:
        parts = key.split(":")
        doc_id = parts[-2]
        question_id = parts[-1]
        
        result_data = r.get(key)
        if result_data:
            doc_scoring_dir = os.path.join(data_dir, "scoring_results", doc_id)
            os.makedirs(doc_scoring_dir, exist_ok=True)
            
            result_file = os.path.join(doc_scoring_dir, f"{question_id}.json")
            try:
                with open(result_file, 'w') as f:
                    json.dump(json.loads(result_data), f)
            except Exception as e:
                log_message(f"Error saving interim scoring results: {e}", status="error", run_id=run_id)

def collect_results(run_id):
    """Collect results from Redis and save to file"""
    r = connect_redis()
    
    log_message(f"Collecting results for run {run_id}", run_id=run_id)
    config = json.loads(r.get(f"benchmark:{run_id}:config"))
    data_dir = config["data_dir"]
    
    # Ensure directories exist
    os.makedirs(os.path.join(data_dir, "processor_results"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "scoring_results"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "logs"), exist_ok=True)
    
    # # Collect all result keys for this run
    # result_keys = r.keys(f"benchmark:{run_id}:result:*")
    
    # results = []
    # for key in result_keys:
    #     result_data = r.get(key)
    #     if result_data:
    #         results.append(json.loads(result_data))
    
    # # Save to file
    # results_file = os.path.join(data_dir, "benchmark_results.jsonl")
    # try:
    #     with open(results_file, "w") as f:
    #         for result in results:
    #             f.write(json.dumps(result) + "\n")
        
    #     log_message(f"Saved {len(results)} results to {results_file}", run_id=run_id)
    # except Exception as e:
    #     log_message(f"Error saving results to {results_file}: {str(e)}", status="error", run_id=run_id)
    #     import traceback
    #     log_message(traceback.format_exc(), status="error", run_id=run_id)
    
    # Save logs
    log_entries = r.lrange("benchmark:logs", 0, -1)
    run_logs = []
    for entry in log_entries:
        log_data = json.loads(entry)
        if "run_id" in log_data and log_data["run_id"] == run_id:
            run_logs.append(log_data)
        elif "message" in log_data and run_id in log_data["message"]:
            run_logs.append(log_data)
    
    # Save logs to file
    logs_file = os.path.join(data_dir, "logs", "benchmark_logs.jsonl")
    try:
        os.makedirs(os.path.dirname(logs_file), exist_ok=True)
        with open(logs_file, "w") as f:
            for log in run_logs:
                f.write(json.dumps(log) + "\n")
        
        log_message(f"Saved {len(run_logs)} log entries to {logs_file}", run_id=run_id)
    except Exception as e:
        log_message(f"Error saving logs to {logs_file}: {str(e)}", status="error", run_id=run_id)
        import traceback
        log_message(traceback.format_exc(), status="error", run_id=run_id)

def test_coordinator():
    """Test function to create a benchmark run and populate the extraction queue"""
    import argparse
    
    # Create a minimal set of arguments for testing
    test_args = argparse.Namespace(
        max_docs=1,  # Process just one document for testing
        dataset_name="default",
        dataset_path="data",
        question_id=None  # No specific question ID for the test
    )
    
    # Set up the benchmark
    run_id = setup_benchmark(test_args)
    
    # Connect to Redis to verify the queue was populated
    r = connect_redis()
    
    # Check if extraction queue has items
    queue_length = r.llen("benchmark:extraction_queue")
    print(f"Extraction queue has {queue_length} items")
    
    # Check if the first item in the queue has the expected structure
    if queue_length > 0:
        task_data = r.lindex("benchmark:extraction_queue", 0)
        task = json.loads(task_data)
        print(f"First task in queue: {json.dumps(task, indent=2)}")
        
        # Verify the task has the expected fields
        assert "run_id" in task, "Task missing run_id"
        assert "doc_id" in task, "Task missing doc_id"
        assert "question_ids" in task, "Task missing question_ids"
        assert isinstance(task["question_ids"], list), "question_ids should be a list"
        
        print("✅ Test passed: Extraction queue populated successfully")
    else:
        print("❌ Test failed: No items in extraction queue")
    
    return run_id

def clear_global_cache(doc_id=None, processor_name=None):
    """
    Clear the global cache. If doc_id is provided, clear only for that document.
    If processor_name is provided, clear only for that processor.
    """
    global_cache_dir = os.path.join("data", "global_cache")
    
    if not os.path.exists(global_cache_dir):
        log_message("Global cache directory does not exist")
        return
    
    if doc_id and processor_name:
        # Clear specific document and processor
        cache_file = os.path.join(global_cache_dir, f"{doc_id}_{processor_name}_result.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            log_message(f"Cleared global cache for document {doc_id} and processor {processor_name}")
        else:
            log_message(f"No cache found for document {doc_id} and processor {processor_name}")
    elif doc_id:
        # Clear all processors for this document
        count = 0
        for file in os.listdir(global_cache_dir):
            if file.startswith(f"{doc_id}_") and file.endswith("_result.json"):
                os.remove(os.path.join(global_cache_dir, file))
                count += 1
        log_message(f"Cleared {count} global cache entries for document {doc_id}")
    elif processor_name:
        # Clear all documents for this processor
        count = 0
        for file in os.listdir(global_cache_dir):
            if f"_{processor_name}_result.json" in file:
                os.remove(os.path.join(global_cache_dir, file))
                count += 1
        log_message(f"Cleared {count} global cache entries for processor {processor_name}")
    else:
        # Clear all cache
        for file in os.listdir(global_cache_dir):
            if file.endswith("_result.json"):
                os.remove(os.path.join(global_cache_dir, file))
        log_message("Cleared all global cache")

def main():
    parser = argparse.ArgumentParser(description='Benchmark Coordinator')
    parser.add_argument('--max-docs', type=int, help='Maximum number of documents to process')
    parser.add_argument('--data-dir', type=str, default='runs', help='Data directory')
    parser.add_argument('--dataset-name', type=str, default='default', help='Dataset name')
    parser.add_argument('--dataset-path', type=str, default='data', help='Dataset path')
    parser.add_argument('--continue-run', type=str, help='Continue an existing run with this run ID')
    parser.add_argument('--test', action='store_true', help='Run a test of the coordinator')
    parser.add_argument('--setup-only', action='store_true', help='Only set up the benchmark without monitoring')
    parser.add_argument('--exit-when-done', action='store_true', help='Exit when benchmark is done (don\'t keep dashboard running)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear the global cache before running')
    parser.add_argument('--doc-id', type=str, help='Process a specific document ID only')
    parser.add_argument('--processor', type=str, help='Process with a specific processor only')
    args = parser.parse_args()
    
    if args.test:
        test_coordinator()
    elif args.continue_run:
        # If continuing an existing run, just monitor it
        run_id = args.continue_run
        print(f"Continuing run: {run_id}")
        monitor_progress(run_id)
    else:
        # Handle cache clearing if requested
        if args.clear_cache:
            clear_global_cache(args.doc_id, args.processor)
            if not args.dataset_name:
                return  # Exit if only clearing cache
        
        # Set up a new benchmark
        run_id = setup_benchmark(args)
        print(f"Benchmark setup complete. Run ID: {run_id}")
        
        # Start monitoring unless --setup-only flag is provided
        if not args.setup_only:
            print(f"Starting monitoring for run: {run_id}")
            monitor_progress(run_id)

if __name__ == "__main__":
    main()
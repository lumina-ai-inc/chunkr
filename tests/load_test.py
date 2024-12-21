import os
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Imports from pyscripts
from api import extract_file, check_task_status
from models import Model, OcrStrategy, SegmentationStrategy, Status, TaskResponse

# Reduce poll interval if messages change quickly on the server
POLL_INTERVAL = 0.4  # seconds between status checks

def main():
    parser = argparse.ArgumentParser(description="Load Testing Script (Requests/Second Prototype)")
    parser.add_argument("--max_pdfs", type=int, default=3, help="Maximum number of PDFs to process")
    parser.add_argument("--requests_per_second", type=float, default=1.0, help="Number of requests to send per second")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run ID for organizing logs")
    args = parser.parse_args()

    # If no run_id is provided, generate one with a timestamp
    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sample configurations
    configs = [
        {"model": Model.HighQuality, "ocr_strategy": OcrStrategy.Auto, "segmentation_strategy": SegmentationStrategy.LayoutAnalysis},
    ]

    runs_folder = Path("runs")
    runs_folder.mkdir(exist_ok=True)

    run_folder = runs_folder / args.run_id
    run_folder.mkdir(exist_ok=True)

    live_log_file = run_folder / "live_pages_sec.log"
    aggregate_log_file = run_folder / "aggregate_run.log"
    status_change_log_file = run_folder / "status_changes.log"

    pdf_folder = Path("input_pdfs")
    pdf_files = sorted(pdf_folder.glob("*.pdf"))[: args.max_pdfs]
    if not pdf_files:
        print("No PDF files found in the input_pdfs folder.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process. Run ID: {args.run_id}")
    print("Beginning load test...")

    def log_live_pages_sec(msg: str):
        with open(live_log_file, "a") as f:
            f.write(msg + "\n")

    def log_aggregate_run(msg: str):
        with open(aggregate_log_file, "a") as f:
            f.write(msg + "\n")

    def log_status_changes(msg: str):
        with open(status_change_log_file, "a") as f:
            f.write(msg + "\n")

    def poll_task_status(task_response: TaskResponse):
        """
        Poll task status every POLL_INTERVAL seconds until it completes/fails.
        Logs *every* time the 'message' or 'status' field changes, plus pages processed,
        and returns (pages_processed, total_duration) for global throughput.
        """
        previous_message = task_response.message or ""
        message_change_times = {previous_message: datetime.now()}

        # Track status changes the same way we were tracking message changes
        previous_status = task_response.status
        status_change_times = {previous_status: datetime.now()}

        pages_processed = 0
        start_time = datetime.now()

        while True:
            time.sleep(POLL_INTERVAL)
            updated_task = check_task_status(task_response.task_url)

            # 1. Log new row if the status changes
            if updated_task.status != previous_status:
                current_time = datetime.now()
                time_since_last_status = (current_time - status_change_times[previous_status]).total_seconds()
                total_elapsed_so_far = (current_time - start_time).total_seconds()

                # Approx pages/sec so far
                avg_pps_so_far = 0
                if total_elapsed_so_far > 0 and updated_task.page_count:
                    avg_pps_so_far = round(updated_task.page_count / total_elapsed_so_far, 2)

                log_status_changes(
                    f"File: {updated_task.file_name}, "
                    f"Status: '{previous_status}' -> '{updated_task.status}', "
                    f"Time Delta: {time_since_last_status:.2f}s, "
                    f"Pages Processed: {updated_task.page_count}, "
                    f"Avg Pages/Sec (so far): {avg_pps_so_far}"
                )

                status_change_times[updated_task.status] = current_time
                previous_status = updated_task.status

            # 2. Log new row if the message changes
            if updated_task.message != previous_message:
                current_time = datetime.now()
                time_since_last_msg = (current_time - message_change_times[previous_message]).total_seconds()
                total_elapsed_so_far = (current_time - start_time).total_seconds()

                # Approx pages/sec up to now
                avg_pps_so_far = 0
                if total_elapsed_so_far > 0 and updated_task.page_count:
                    avg_pps_so_far = round(updated_task.page_count / total_elapsed_so_far, 2)

                log_status_changes(
                    f"File: {updated_task.file_name}, "
                    f"Message: '{previous_message}' -> '{updated_task.message}', "
                    f"Time Delta: {time_since_last_msg:.2f}s, "
                    f"Pages Processed: {updated_task.page_count}, "
                    f"Avg Pages/Sec (so far): {avg_pps_so_far}"
                )

                message_change_times[updated_task.message] = current_time
                previous_message = updated_task.message

            # Record pages processed in a live log
            if updated_task.page_count and updated_task.page_count != pages_processed:
                pages_processed = updated_task.page_count
                log_live_pages_sec(
                    f"Time: {datetime.now().isoformat()}, File: {updated_task.file_name}, "
                    f"Pages Processed: {pages_processed}"
                )

            # Check if done
            if updated_task.status in [Status.Succeeded, Status.Failed, Status.Canceled]:
                finish_time = datetime.now()
                total_duration = (finish_time - start_time).total_seconds()
                file_pages_per_second = (
                    round(pages_processed / total_duration, 2) if total_duration > 0 else 0
                )

                log_aggregate_run(
                    f"File: {updated_task.file_name}, Final Status: {updated_task.status}, "
                    f"Total Pages: {pages_processed}, Elapsed: {total_duration:.2f}s, "
                    f"Avg Pages/Sec: {file_pages_per_second}"
                )
                return pages_processed, total_duration

    # Track total run for net throughput across all files
    run_start_time = datetime.now()
    total_pages_processed_across_all = 0
    futures = []

    with ThreadPoolExecutor() as executor:
        for i, pdf in enumerate(pdf_files):
            chosen_config = random.choice(configs)
            print(f"\nSending request for file: {pdf.name} with config: {chosen_config}")

            task_response = extract_file(
                file_to_send=str(pdf),
                model=chosen_config["model"],
                ocr_strategy=chosen_config["ocr_strategy"],
                segmentation_strategy=chosen_config["segmentation_strategy"]
            )

            future = executor.submit(poll_task_status, task_response)
            futures.append(future)

            # Throttle requests if desired
            if args.requests_per_second > 0 and i < len(pdf_files) - 1:
                time.sleep(1.0 / args.requests_per_second)

        # If a file fails, keep going
        for future in as_completed(futures):
            try:
                pages_for_file, _duration_for_file = future.result()
                total_pages_processed_across_all += pages_for_file
            except Exception as exc:
                log_aggregate_run(f"Error while polling a task: {str(exc)}")

    # COMPUTE NET THROUGHPUT ACROSS THE ENTIRE RUN
    run_end_time = datetime.now()
    run_duration = (run_end_time - run_start_time).total_seconds()
    net_throughput = 0
    if run_duration > 0:
        net_throughput = round(total_pages_processed_across_all / run_duration, 2)

    summary_msg = (
        f"ALL FILES COMPLETE. Net throughput: {net_throughput} pages/sec, "
        f"Total pages across all files: {total_pages_processed_across_all}, "
        f"Total run time: {round(run_duration, 2)}s"
    )
    log_aggregate_run(summary_msg)
    print(summary_msg)

    print(f"Load test complete. Logs stored in: {run_folder}")

if __name__ == "__main__":
    main()
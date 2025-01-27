import pydantic

class ProcessPayload(pydantic.BaseModel):
    run_id: str
    input_file: str
    output_dir: str
    start_time: str
    log_dir: str

class WritePayload(pydantic.BaseModel):
    run_id: str
    log_dir: str
    task_path: str
    start_time: str
    end_time: str

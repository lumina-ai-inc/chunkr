import pydantic

class ProcessPayload(pydantic.BaseModel):
    input_file: str
    output_dir: str
    start_time: str
    stats_path: str

class WritePayload(pydantic.BaseModel):
    stats_path: str
    task_path: str
    start_time: str
    end_time: str
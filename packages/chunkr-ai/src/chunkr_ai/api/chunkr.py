
from .models import TaskResponse, OcrStrategy, SegmentationStrategy, Configuration

class Chunkr:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self):
        return {"Authorization": self.api_key}

    def upload_file_async(self, file_path: str, config: Configuration = None) -> TaskResponse:
        import requests, os
        url = f"{self.base_url}/api/v1/task"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            r = requests.post(url, files=files, json=config.dict(), headers=self._headers())
            r.raise_for_status()
            return TaskResponse(**r.json())

    def poll_task(self, task_url: str) -> TaskResponse:
        import requests, time
        while True:
            r = requests.get(task_url, headers=self._headers())
            r.raise_for_status()
            task = TaskResponse(**r.json())
            if task.status not in ("Starting", "Processing"):
                return task
            time.sleep(1)

    def upload_file_sync(self, file_path: str, config: Configuration = None) -> TaskResponse:
        task = self.upload_file_async(file_path, config)
        if not task.task_url:
            return task
        return self.poll_task(task.task_url)

    def html(self, task_response: TaskResponse) -> str:
        if not task_response.output:
            return ""
        parts = []
        for c in task_response.output.chunks:
            for s in c.segments:
                if s.html:
                    parts.append(s.html)
        return "\n".join(parts)

    def markdown(self, task_response: TaskResponse) -> str:
        if not task_response.output:
            return ""
        parts = []
        for c in task_response.output.chunks:
            for s in c.segments:
                if s.markdown:
                    parts.append(s.markdown)
        return "\n".join(parts)



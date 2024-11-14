from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
from collections import deque
import time
from typing import List

app = FastAPI()

# Queue to store pending tasks
pending_tasks = deque()
# Lock to protect batch processing
processing_lock = asyncio.Lock()
# Event to signal batch processor
batch_event = asyncio.Event()

class Task(BaseModel):
    data: str

class TaskResponse(BaseModel):
    result: str

async def process_tasks(tasks: List[Task]) -> List[TaskResponse]:
    # Your batch processing logic here
    # This is just an example
    return [TaskResponse(result=f"Processed: {task.data}") for task in tasks]

async def batch_processor():
    while True:
        # Wait for tasks to arrive
        await batch_event.wait()
        batch_event.clear()
        
        # Wait additional 500ms for more tasks
        await asyncio.sleep(0.5)
        
        async with processing_lock:
            if not pending_tasks:
                continue
                
            # Get all pending tasks
            current_batch = list(pending_tasks)
            pending_tasks.clear()
            
        # Process the batch
        results = await process_tasks(current_batch)
        
        # Resolve all waiting futures
        for task, result in zip(current_batch, results):
            task.future.set_result(result)

@app.on_event("startup")
async def startup_event():
    # Start the batch processor
    asyncio.create_task(batch_processor())

@app.post("/task", response_model=TaskResponse)
async def create_task(task: Task):
    # Create a future to get the result
    future = asyncio.Future()
    
    # Add task metadata
    task.future = future
    
    # Add task to queue
    pending_tasks.append(task)
    
    # Signal batch processor
    batch_event.set()
    
    # Wait for result
    result = await future
    return result
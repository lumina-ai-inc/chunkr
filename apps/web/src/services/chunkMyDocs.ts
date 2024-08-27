import { UploadForm } from "../models/upload.model";
import { Status, TaskResponse } from "../models/task.model";
import { uploadFile, getTask, getFile } from "./uploadFileApi";

export async function uploadFileStep(
  payload: UploadForm
): Promise<TaskResponse> {
  return await uploadFile(payload);
}

export async function checkTaskStatus(taskId: string): Promise<TaskResponse> {
  let taskResponse: TaskResponse;
  do {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    taskResponse = await getTask(taskId);
    console.log("Task Response:", taskResponse);
  } while (
    taskResponse.status !== Status.Succeeded &&
    taskResponse.status !== Status.Failed
  );
  return taskResponse;
}

export function handleTaskStatus(taskResponse: TaskResponse): void {
  if (taskResponse.status === Status.Failed) {
    throw new Error(`Task failed: ${taskResponse.message}`);
  }
}

export async function retrieveFileContent(fileUrl: string): Promise<string> {
  return await getFile(fileUrl);
}

export async function processFileUpload(payload: UploadForm): Promise<string> {
  try {
    // Step 1: Upload the file
    const initialResponse = await uploadFileStep(payload);

    // Step 2: Check task status
    const finalTaskResponse = await checkTaskStatus(initialResponse.task_id);

    // Step 3: Handle the final status
    handleTaskStatus(finalTaskResponse);

    // Step 4: Retrieve the file content
    if (finalTaskResponse.file_url) {
      return await retrieveFileContent(finalTaskResponse.file_url);
    } else {
      throw new Error("File URL not provided in the successful response");
    }
  } catch (error) {
    console.error("Error processing file upload:", error);
    throw error;
  }
}

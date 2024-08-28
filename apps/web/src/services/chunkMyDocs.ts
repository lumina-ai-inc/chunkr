import { UploadForm } from "../models/upload.model";
import { Status, TaskResponse } from "../models/task.model";
import { uploadFile, getFile, getPDF } from "./uploadFileApi";
import { BoundingBoxes } from "../models/chunk.model";

export async function uploadFileStep(
  payload: UploadForm
): Promise<TaskResponse> {
  return await uploadFile(payload);
}

export function handleTaskStatus(taskResponse: TaskResponse): void {
  if (taskResponse.status === Status.Failed) {
    throw new Error(`Task failed: ${taskResponse.message}`);
  }
}

export async function retrieveFileContent(
  fileUrl: string
): Promise<BoundingBoxes> {
  const fileContent = await getFile(fileUrl);
  return fileContent;
}

export async function fetchPdfFile(fileUrl: string): Promise<File> {
  const fileContent = await getPDF(fileUrl);
  return fileContent;
}

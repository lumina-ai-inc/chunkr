import { UploadForm } from "../models/upload.model";
import { Status } from "../models/task.model";
import { uploadFile, getTask, getFile } from "./uploadFileApi";

export async function processFileUpload(payload: UploadForm): Promise<string> {
  try {
    // Step 1: Upload the file
    const initialResponse = await uploadFile(payload);
    let taskResponse = initialResponse;

    // Step 2: Check task status in a loop
    while (
      taskResponse.status !== Status.Succeeded &&
      taskResponse.status !== Status.Failed
    ) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      console.log("Task Response:", taskResponse);
      taskResponse = await getTask(taskResponse.task_id);
    }

    // Step 3: Handle the final status
    if (taskResponse.status === Status.Failed) {
      throw new Error(`Task failed: ${taskResponse.message}`);
    }

    // Step 4: Retrieve the file content
    if (taskResponse.file_url) {
      return await getFile(taskResponse.file_url);
    } else {
      throw new Error("File URL not provided in the successful response");
    }
  } catch (error) {
    console.error("Error processing file upload:", error);
    throw error;
  }
}

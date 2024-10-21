import { makeRequest, pollTask } from "../api.js";
import pLimit from "p-limit";
import { MAX_CONCURRENT_REQUESTS } from "../config.js";
import { TaskConfig, TaskResponse } from "../models.js";

const limit = pLimit(MAX_CONCURRENT_REQUESTS);

export async function processTask(
  config: TaskConfig,
  filePath: string
): Promise<TaskResponse> {
  const task = await makeRequest(config, filePath);
  if (!task) throw new Error("Failed to make request");

  console.log(`Making request for file: ${filePath}`);
  console.log(`API Base URL: ${process.env.API_BASE_URL}`);
  console.log(`API Key: ${process.env.API_KEY ? "Set" : "Not set"}`);

  let currentTask = await pollTask(task.task_id);

  while (
    currentTask.status !== "Succeeded" &&
    currentTask.status !== "Failed"
  ) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    currentTask = await pollTask(task.task_id);
  }

  console.log(
    `Task ${currentTask.task_id} finished with status: ${currentTask.status}`
  );

  return currentTask;
}

export async function runBatch(
  config: TaskConfig,
  files: string[]
): Promise<TaskResponse[]> {
  console.log(`Starting batch processing for ${files.length} files`);
  const promises = files.map((file) => limit(() => processTask(config, file)));
  const results = await Promise.all(promises);
  console.log(`Completed batch processing for ${files.length} files`);
  return results;
}

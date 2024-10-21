import { makeRequest, pollTask } from "../api.js";
import pLimit from "p-limit";
import { MAX_CONCURRENT_REQUESTS } from "../config.js";
import { TaskConfig } from "../models.js";

const limit = pLimit(MAX_CONCURRENT_REQUESTS);

export async function processTask(
  config: TaskConfig,
  filePath: string
): Promise<void> {
  const task = await makeRequest(config, filePath);
  if (!task) return;

  while (true) {
    const currentTask = await pollTask(task.task_id);

    if (currentTask.status === "Succeeded" || currentTask.status === "Failed") {
      console.log(
        `Task ${currentTask.task_id} finished with status: ${currentTask.status}`
      );
      break;
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

export async function runBatch(
  config: TaskConfig,
  files: string[]
): Promise<void> {
  console.log(`Starting batch processing for ${files.length} files`);
  const promises = files.map((file) => limit(() => processTask(config, file)));
  await Promise.all(promises);
  console.log(`Completed batch processing for ${files.length} files`);
}

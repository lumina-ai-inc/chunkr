import { Worker, isMainThread, workerData, parentPort } from "worker_threads";
import { runBatch } from "../batch/batchProcessor.js";
import { createCsvWriter } from "./csvWriter.js";
import {
  INPUT_FOLDER,
  MAX_FILES_TO_PROCESS,
  REQUESTS_PER_SECOND,
} from "../config.js";
import fs from "fs/promises";
import path from "path";
import { TaskConfig, ModelOCRStrategy } from "../models.js";
import { pollTask } from "../api.js";

const NUM_WORKERS = 4;

const CONFIGS: TaskConfig[] = [
  { model: "HighQuality", ocr_strategy: "Auto", target_chunk_length: 512 },
  { model: "HighQuality", ocr_strategy: "All", target_chunk_length: 512 },
  { model: "Fast", ocr_strategy: "Auto", target_chunk_length: 512 },
  { model: "Fast", ocr_strategy: "Off", target_chunk_length: 512 },
];

function getRandomConfig(): TaskConfig {
  const rand = Math.random();
  if (rand < 0.65) {
    return CONFIGS[0]; // HighQuality, Auto
  } else if (rand < 0.8) {
    return CONFIGS[1]; // HighQuality, All
  } else if (rand < 0.9) {
    return CONFIGS[2]; // Fast, Auto
  } else {
    return CONFIGS[3]; // Fast, Off
  }
}

async function runLoadTest(files: string[]): Promise<void> {
  const csvWriters = new Map<
    ModelOCRStrategy,
    ReturnType<typeof createCsvWriter>
  >();

  for (const file of files) {
    const config = getRandomConfig();
    const modelOCRStrategy: ModelOCRStrategy = `${config.model}_${config.ocr_strategy}`;

    if (!csvWriters.has(modelOCRStrategy)) {
      csvWriters.set(modelOCRStrategy, createCsvWriter(modelOCRStrategy));
    }

    const csvWriter = csvWriters.get(modelOCRStrategy)!;
    const overallStartTime = new Date().toISOString();

    console.log(`Starting task for file: ${file}`);
    const taskResponses = await runBatch(config, [file]);

    for (const taskResponse of taskResponses) {
      let currentTask = taskResponse;
      let lastMessage = "";
      let stageStartTime = new Date().toISOString();

      while (true) {
        if (currentTask.message !== lastMessage) {
          if (lastMessage !== "") {
            // Log the completed stage
            const stageEndTime = new Date().toISOString();
            await csvWriter.writeRecords([
              {
                file_name: file,
                start_time: stageStartTime,
                end_time: stageEndTime,
                model: config.model,
                ocr_strategy: config.ocr_strategy,
                target_chunk_length: config.target_chunk_length,
                task_id: currentTask.task_id,
                status: currentTask.status,
                message: lastMessage,
              },
            ]);
            console.log(
              `Task ${currentTask.task_id} - Completed stage: ${lastMessage}`
            );
          }
          // Start new stage
          lastMessage = currentTask.message;
          stageStartTime = new Date().toISOString();
        }

        if (
          currentTask.status === "Succeeded" ||
          currentTask.status === "Failed"
        ) {
          break;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000)); // Poll every second
        currentTask = await pollTask(currentTask.task_id);
      }

      // Log final state
      const overallEndTime = new Date().toISOString();
      await csvWriter.writeRecords([
        {
          file_name: file,
          start_time: stageStartTime,
          end_time: overallEndTime,
          model: config.model,
          ocr_strategy: config.ocr_strategy,
          target_chunk_length: config.target_chunk_length,
          task_id: currentTask.task_id,
          status: currentTask.status,
          message: currentTask.message,
        },
      ]);
      console.log(
        `Task ${currentTask.task_id} completed - Final Status: ${currentTask.status}, Message: ${currentTask.message}`
      );
    }

    const interval = 1000 / REQUESTS_PER_SECOND;
    const elapsedTime =
      new Date().getTime() - new Date(overallStartTime).getTime();
    const delay = Math.max(0, interval - elapsedTime);
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
}

async function startLoadTest(): Promise<void> {
  if (isMainThread) {
    console.log("[Main] Starting load test...");
    try {
      const files = await fs.readdir(INPUT_FOLDER);
      const filePaths = files
        .map((file) => path.join(INPUT_FOLDER, file))
        .slice(0, MAX_FILES_TO_PROCESS);

      console.log(`[Main] Found ${filePaths.length} files to process`);

      const filesPerWorker = Math.ceil(filePaths.length / NUM_WORKERS);
      let completedWorkers = 0;

      for (let i = 0; i < NUM_WORKERS; i++) {
        const workerFiles = filePaths.slice(
          i * filesPerWorker,
          (i + 1) * filesPerWorker
        );
        const worker = new Worker(new URL(import.meta.url), {
          workerData: { files: workerFiles },
        });

        worker.on("error", (error) => {
          console.error(`[Main] Worker ${i + 1} error:`, error);
        });

        worker.on("exit", (code) => {
          completedWorkers++;
          console.log(
            `[Main] Worker ${i + 1} completed. Total completed: ${completedWorkers}`
          );
          if (completedWorkers === NUM_WORKERS) {
            console.log("[Main] All workers completed. Exiting...");
            process.exit(0);
          }
        });
      }
    } catch (error) {
      console.error("[Main] Error in startLoadTest:", error);
    }
  } else {
    try {
      if (!workerData || !workerData.files) {
        throw new Error("Worker data or files are missing");
      }
      await runLoadTest(workerData.files);
    } catch (error) {
      console.error(`[Worker] Error in worker:`, error);
    } finally {
      if (parentPort) {
        parentPort.postMessage("Worker finished");
      }
    }
  }
}

if (!isMainThread) {
  startLoadTest().catch((error) => {
    console.error(`[Worker] Unhandled error in worker:`, error);
  });
}

export { startLoadTest };
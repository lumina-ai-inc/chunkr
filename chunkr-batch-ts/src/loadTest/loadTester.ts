import { Worker, isMainThread, workerData } from "worker_threads";
import { runBatch } from "../batch/batchProcessor.js";
import { createCsvWriter } from "./csvWriter.js";
import {
  INPUT_FOLDER,
  REQUESTS_PER_SECOND,
  MAX_FILES_TO_PROCESS,
} from "../config.js";
import fs from "fs/promises";
import path from "path";
import { TaskConfig, ModelOCRStrategy } from "../models.js";

const DEFAULT_CONFIG: TaskConfig = {
  model: "HighQuality",
  ocr_strategy: "Auto",
  target_chunk_length: 512,
};

export async function runLoadTest(
  config: TaskConfig,
  files: string[]
): Promise<void> {
  const csvWriter = createCsvWriter(
    `${config.model}_${config.ocr_strategy}` as ModelOCRStrategy
  );
  const interval = 1000 / REQUESTS_PER_SECOND;

  const logMessage = (message: string) => {
    if (isMainThread) {
      console.log(message);
    } else {
      process.send!(message);
    }
  };

  logMessage(`Starting load test for config: ${JSON.stringify(config)}`);
  logMessage(`Processing ${files.length} files`);

  for (const file of files) {
    const startTime = Date.now();

    try {
      await runBatch(config, [file]);
      logMessage(`Processed file: ${file}`);

      await csvWriter.writeRecords([
        {
          file_name: file,
          start_time: new Date(startTime).toISOString(),
          end_time: new Date().toISOString(),
          model: config.model,
          ocr_strategy: config.ocr_strategy,
          target_chunk_length: config.target_chunk_length,
        },
      ]);
    } catch (error) {
      logMessage(`Error processing file ${file}: ${error}`);
    }

    const elapsedTime = Date.now() - startTime;
    const delay = Math.max(0, interval - elapsedTime);
    await new Promise((resolve) => setTimeout(resolve, delay));
  }

  logMessage(`Completed load test for config: ${JSON.stringify(config)}`);
}

export async function startLoadTest(): Promise<void> {
  const files = await fs.readdir(INPUT_FOLDER);
  const filePaths = files.map((file) => path.join(INPUT_FOLDER, file));
  console.log(`Input folder: ${INPUT_FOLDER}`);
  console.log(`Number of files found: ${files.length}`);

  // Limit the total number of files to MAX_FILES_TO_PROCESS
  const limitedFilePaths = filePaths.slice(0, MAX_FILES_TO_PROCESS);

  const configurations: TaskConfig[] = [
    { ...DEFAULT_CONFIG },
    { ...DEFAULT_CONFIG, model: "Fast" },
    { ...DEFAULT_CONFIG, ocr_strategy: "All" },
    { ...DEFAULT_CONFIG, ocr_strategy: "Off" },
    { ...DEFAULT_CONFIG, model: "Fast", ocr_strategy: "All" },
    { ...DEFAULT_CONFIG, model: "Fast", ocr_strategy: "Off" },
  ];

  if (isMainThread) {
    console.log(
      `Starting load test with multiple configurations (max ${MAX_FILES_TO_PROCESS} files)...`
    );

    const workerPromises = configurations.map((config) => {
      return new Promise<void>((resolve, reject) => {
        const worker = new Worker(new URL(import.meta.url), {
          workerData: {
            config,
            files: limitedFilePaths,
          },
        });

        worker.on("message", (message) => {
          console.log(message);
        });

        worker.on("error", reject);
        worker.on("exit", (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Worker stopped with exit code ${code}`));
          }
        });
      });
    });

    try {
      await Promise.all(workerPromises);
      console.log("Load test completed for all configurations. Exiting...");
    } catch (error) {
      console.error("Error during load test:", error);
    }
  } else {
    // This is the worker thread
    const { config, files } = workerData;
    await runLoadTest(config, files);
    process.exit(0);
  }
}

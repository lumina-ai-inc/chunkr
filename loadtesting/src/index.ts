import { Worker, isMainThread, workerData, parentPort } from "worker_threads";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import dotenv from "dotenv";
import { createObjectCsvWriter } from "csv-writer";
import {
  TaskResponse,
  AggregateResults,
  WorkerResult,
  FailureTypes,
} from "./models.js";
import pLimit from "p-limit";
import { fileURLToPath } from "url";
import path from "path";
import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";
import { performance } from "perf_hooks";
import { PDFDocument } from "pdf-lib";

dotenv.config();

const API_URL = process.env.API_URL as string;
const API_KEY = process.env.API_KEY as string;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
if (!API_KEY || !API_URL) {
  console.error("API_KEY or API_URL not found in environment variables");
  process.exit(1);
}

const eventEmitter = new EventEmitter();

const MAX_FILES_TO_PROCESS = 1000; // Adjust this value as needed
const REQUESTS_PER_SECOND = 50; // You can adjust this value
const WORKERS_PER_CONFIG = 4; // Adjust this number as needed
const INPUT_FOLDER = path.join(__dirname, "..", "input");
const OUTPUT_FOLDER = path.join(__dirname, "..", "output");
const RUN_ID = `${new Date().toISOString().replace(/[:.]/g, "-")}_${uuidv4().slice(0, 8)}`;
let RUN_FOLDER = path.join(OUTPUT_FOLDER, RUN_ID);

// Ensure run folder exists
if (!fs.existsSync(RUN_FOLDER)) {
  fs.mkdirSync(RUN_FOLDER, { recursive: true });
}

type ModelConfig = {
  model: "HighQuality" | "Fast";
  ocrStrategy: "All" | "Auto";
  percentage: number;
};

// Modify the MODEL_CONFIGS to include the number of workers
const MODEL_CONFIGS: (ModelConfig & { workers: number })[] = [
  {
    model: "HighQuality",
    ocrStrategy: "Auto",
    percentage: 100,
    workers: WORKERS_PER_CONFIG,
  },
];

// Save configurations to a txt file
const configFilePath = path.join(RUN_FOLDER, "config.txt");
const configData = {
  MODEL_CONFIGS,
  MAX_FILES_TO_PROCESS,
  REQUESTS_PER_SECOND,
};
fs.writeFileSync(configFilePath, JSON.stringify(configData, null, 2));

function createCsvWriter(
  model: "HighQuality" | "Fast",
  ocrStrategy: "All" | "Auto",
  type: "progress"
) {
  const configFolder = path.join(
    RUN_FOLDER,
    `${model.toLowerCase()}_${ocrStrategy.toLowerCase()}`
  );
  if (!fs.existsSync(configFolder)) {
    fs.mkdirSync(configFolder, { recursive: true });
  }

  const fileName = "task_progress.csv";
  const filePath = path.join(configFolder, fileName);

  const header = [
    { id: "task_id", title: "task_id" },
    { id: "file_name", title: "file_name" },
    { id: "page_count", title: "page_count" },
    { id: "message", title: "message" },
    { id: "start_time", title: "start_time" },
    { id: "end_time", title: "end_time" },
    { id: "duration_ms", title: "duration_ms" },
  ];

  return createObjectCsvWriter({
    path: filePath,
    header: header,
    append: true,
  });
}

async function makeRequest(filePath: string, config: ModelConfig) {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await PDFDocument.load(dataBuffer);

    console.log(
      `File ${path.basename(filePath)} has ${data.getPageCount()} pages`
    ); // Temporary log for testing

    if (data.getPageCount() > 500) {
      console.log(
        `Skipping file ${path.basename(filePath)} - more than 500 pages`
      );
      return null;
    }

    const form = new FormData();
    form.append("file", fs.createReadStream(filePath));
    form.append("model", config.model);
    form.append("target_chunk_length", "512");
    form.append("ocr_strategy", config.ocrStrategy);

    try {
      const response = await axios.post<TaskResponse>(API_URL, form, {
        headers: {
          ...form.getHeaders(),
          Authorization: API_KEY,
        },
      });
      return response.data;
    } catch (error) {
      return null;
    }
  } catch (error) {
    return null;
  }
}

async function pollTask(
  taskId: string,
  config: ModelConfig
): Promise<TaskResponse | null> {
  console.log(
    `Starting task ${taskId} for model ${config.model} with OCR strategy ${config.ocrStrategy}`
  );
  const progressCsvWriter = createCsvWriter(
    config.model,
    config.ocrStrategy,
    "progress"
  );
  let lastMessage = "";
  let messageStartTime = new Date().toISOString();
  let taskStartTime = new Date().toISOString();

  while (true) {
    try {
      const response = await axios.get<TaskResponse>(`${API_URL}/${taskId}`, {
        headers: { Authorization: API_KEY },
      });
      const task = response.data;
      const currentTime = new Date().toISOString();

      if (task.message !== lastMessage) {
        if (lastMessage !== "") {
          const startDate = new Date(messageStartTime);
          const endDate = new Date(currentTime);
          const durationMs = endDate.getTime() - startDate.getTime();

          await progressCsvWriter.writeRecords([
            {
              task_id: task.task_id,
              file_name: task.file_name,
              page_count: task.page_count,
              message: lastMessage,
              start_time: messageStartTime,
              end_time: currentTime,
              duration_ms: durationMs,
            },
          ]);
        }

        lastMessage = task.message;
        messageStartTime = currentTime;
      }

      if (task.status === "Succeeded" || task.status === "Failed") {
        const startDate = new Date(taskStartTime);
        const endDate = new Date(currentTime);
        const durationMs = endDate.getTime() - startDate.getTime();

        await progressCsvWriter.writeRecords([
          {
            task_id: task.task_id,
            file_name: task.file_name,
            page_count: task.page_count,
            message: lastMessage,
            start_time: messageStartTime,
            end_time: currentTime,
            duration_ms: durationMs,
          },
        ]);

        console.log(`Task ${taskId} finished with status: ${task.status}`);
        console.log(`Page count for task ${taskId}: ${task.page_count}`);
        return task;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    } catch (error) {
      console.error(`Error polling task ${taskId}:`, error);
      break;
    }
  }

  return null;
}

// Add this new function to distribute files among configs
function distributeFiles(
  files: string[],
  configs: (ModelConfig & { workers: number })[]
): Map<string, string[]> {
  const distribution = new Map<string, string[]>();
  let fileIndex = 0;

  configs.forEach((config) => {
    const configKey = `${config.model}_${config.ocrStrategy}`;
    const filesToProcess = Math.floor(
      (MAX_FILES_TO_PROCESS * config.percentage) / 100
    );
    distribution.set(configKey, []);

    for (let i = 0; i < filesToProcess && fileIndex < files.length; i++) {
      distribution.get(configKey)!.push(files[fileIndex]);
      fileIndex++;
    }
  });

  return distribution;
}

async function runLoadTest(
  config: ModelConfig,
  assignedFiles: string[]
): Promise<{ totalPages: number; failureTypes: FailureTypes }> {
  let processedFiles = 0;
  let totalPages = 0;
  let failureTypes: FailureTypes = {
    startTaskFailed: 0,
    pollTaskFailed: 0,
    taskStatusFailed: 0,
  };

  const limit = pLimit(REQUESTS_PER_SECOND);

  const tasks = assignedFiles.map((file) => {
    return limit(async () => {
      const filePath = path.join(INPUT_FOLDER, file);
      const task = await makeRequest(filePath, config);

      if (task) {
        const pollResult = await pollTask(task.task_id, config);
        if (pollResult) {
          if (pollResult.status === "Succeeded") {
            processedFiles++;
            totalPages += pollResult.page_count;
            console.log(
              `Processed file ${file} with ${pollResult.page_count} pages`
            );
            // Send message to main thread for each processed page
            for (let i = 0; i < pollResult.page_count; i++) {
              parentPort?.postMessage({ type: "pageProcessed" });
            }
          } else if (pollResult.status === "Failed") {
            console.error(
              `Task failed for file ${file}: ${pollResult.message}`
            );
            failureTypes.taskStatusFailed++;
          }
        } else {
          console.error(`Failed to poll task for file ${file}`);
          failureTypes.pollTaskFailed++;
        }
      } else {
        console.error(`Failed to start task for file ${file}`);
        failureTypes.startTaskFailed++;
      }
    });
  });

  await Promise.all(tasks);

  // Write results to a text file
  const resultFilePath = path.join(
    RUN_FOLDER,
    `${config.model}_${config.ocrStrategy}_results.txt`
  );
  const resultContent = `
Total pages processed: ${totalPages}
Failed to start task: ${failureTypes.startTaskFailed}
Failed to poll task: ${failureTypes.pollTaskFailed}
Tasks completed with failure status: ${failureTypes.taskStatusFailed}
Total failed files: ${failureTypes.startTaskFailed + failureTypes.pollTaskFailed + failureTypes.taskStatusFailed}
  `.trim();

  fs.writeFileSync(resultFilePath, resultContent);

  return { totalPages, failureTypes };
}

// Add these new variables and functions
const AGGREGATE_LOG_INTERVAL = 5000; // 5 seconds
const aggregateLogPath = path.join(RUN_FOLDER, "aggregate_log.txt");
let totalProcessedPages = 0;
let runStartTime: number;

function initializeAggregateLog() {
  runStartTime = performance.now();
  fs.writeFileSync(aggregateLogPath, "Time (s),Pages Processed,Pages/Second\n");
}

function updateAggregateLog() {
  const currentTime = performance.now();
  const elapsedSeconds = (currentTime - runStartTime) / 1000;
  const pagesPerSecond = totalProcessedPages / elapsedSeconds;

  const logEntry = `${elapsedSeconds.toFixed(2)},${totalProcessedPages},${pagesPerSecond.toFixed(2)}\n`;
  fs.appendFileSync(aggregateLogPath, logEntry);
}

if (isMainThread) {
  // Remove the ensureDirectoryExists call
  const files = fs
    .readdirSync(INPUT_FOLDER)
    .filter((file) => file.endsWith(".pdf"));

  const fileDistribution = distributeFiles(files, MODEL_CONFIGS);

  // Calculate the total number of workers
  const numWorkers = MODEL_CONFIGS.reduce(
    (sum, config) => sum + config.workers,
    0
  );
  console.log(`Starting ${numWorkers} workers...`);
  let completedWorkers = 0;

  const workerResults: WorkerResult[] = [];

  initializeAggregateLog();
  const aggregateLogInterval = setInterval(
    updateAggregateLog,
    AGGREGATE_LOG_INTERVAL
  );

  MODEL_CONFIGS.forEach((config) => {
    const configKey = `${config.model}_${config.ocrStrategy}`;
    const assignedFiles = fileDistribution.get(configKey) || [];

    // Distribute files evenly among workers for this config
    const filesPerWorker = Math.ceil(assignedFiles.length / config.workers);

    for (let i = 0; i < config.workers; i++) {
      const workerFiles = assignedFiles.slice(
        i * filesPerWorker,
        (i + 1) * filesPerWorker
      );

      const worker = new Worker(__filename, {
        workerData: { config, assignedFiles: workerFiles, RUN_FOLDER },
      });

      worker.on("message", (message: { type: string; data: WorkerResult }) => {
        if (message.type === "workerComplete") {
          workerResults.push(message.data);
          updateWorkerResultFile(config, message.data);
          completedWorkers++;
          if (completedWorkers === numWorkers) {
            eventEmitter.emit("allWorkersComplete");
          }
        } else if (message.type === "pageProcessed") {
          totalProcessedPages++;
        }
      });

      worker.on("error", (error) => {
        console.error(`Worker error: ${error}`);
      });

      worker.on("exit", (code) => {
        if (code !== 0) {
          console.error(`Worker stopped with exit code ${code}`);
        }
      });
    }
  });

  eventEmitter.on("allWorkersComplete", () => {
    clearInterval(aggregateLogInterval);
    updateAggregateLog(); // Final update
    const aggregateResults = calculateAggregateResults(workerResults);
    updateConfigFile(aggregateResults);
    console.log("Load test completed for all configurations. Exiting...");
    process.exit(0);
  });
} else {
  // This code runs in worker threads
  const { config, assignedFiles, RUN_FOLDER: workerRunFolder } = workerData;
  RUN_FOLDER = workerRunFolder;
  const startTime = performance.now();
  runLoadTest(config, assignedFiles).then(({ totalPages, failureTypes }) => {
    const endTime = performance.now();
    parentPort?.postMessage({
      type: "workerComplete",
      data: { totalPages, failureTypes, startTime, endTime },
    });
  });
}

function calculateAggregateResults(results: WorkerResult[]): AggregateResults {
  const totalTime =
    Math.max(...results.map((r) => r.endTime)) -
    Math.min(...results.map((r) => r.startTime));
  const totalPages = results.reduce((sum, r) => sum + r.totalPages, 0);
  const pagesPerSecond = totalPages / (totalTime / 1000);

  return {
    totalTime,
    totalPages,
    pagesPerSecond,
  };
}

function updateConfigFile(results: AggregateResults) {
  const configFilePath = path.join(RUN_FOLDER, "config.txt");
  const configData = JSON.parse(fs.readFileSync(configFilePath, "utf-8"));

  configData.aggregateResults = results;

  fs.writeFileSync(configFilePath, JSON.stringify(configData, null, 2));
}

function updateWorkerResultFile(
  config: ModelConfig,
  result: WorkerResult & { failureTypes: FailureTypes }
) {
  const resultFilePath = path.join(
    RUN_FOLDER,
    `${config.model}_${config.ocrStrategy}_results.txt`
  );
  const duration = (result.endTime - result.startTime) / 1000; // Convert to seconds
  const pagesPerSecond = result.totalPages / duration;
  const totalFailedFiles =
    result.failureTypes.startTaskFailed +
    result.failureTypes.pollTaskFailed +
    result.failureTypes.taskStatusFailed;

  const content = `
Total pages processed: ${result.totalPages}
Failed to start task: ${result.failureTypes.startTaskFailed}
Failed to poll task: ${result.failureTypes.pollTaskFailed}
Tasks completed with failure status: ${result.failureTypes.taskStatusFailed}
Total failed files: ${totalFailedFiles}
Duration: ${duration.toFixed(2)} seconds
Pages per second: ${pagesPerSecond.toFixed(2)}
  `.trim();

  fs.writeFileSync(resultFilePath, content);
}

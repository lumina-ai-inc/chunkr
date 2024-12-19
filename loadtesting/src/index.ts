import { Worker, isMainThread, workerData, parentPort } from "worker_threads";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import dotenv from "dotenv";
import { createObjectCsvWriter } from "csv-writer";
import pLimit from "p-limit";
import { fileURLToPath } from "url";
import path from "path";
import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";
import { performance } from "perf_hooks";
import { PDFDocument } from "pdf-lib";

import {
  TaskResponse,
  AggregateResults,
  WorkerResult,
  FailureTypes,
  ModelConfig,
  Model,
  OcrStrategy,
} from "./models";

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

const MAX_FILES_TO_PROCESS = 20; // Adjust this value as needed
const CONCURRENT_REQUESTS_PER_WORKER = 20; // You can adjust this value
const WORKERS_PER_CONFIG = 1; // Adjust this number as needed
const INPUT_FOLDER = path.join(__dirname, "..", "input");
const OUTPUT_FOLDER = path.join(__dirname, "..", "output");
const RUN_ID = `${new Date().toISOString().replace(/[:.]/g, "-")}_${uuidv4().slice(0, 8)}`;
let RUN_FOLDER = path.join(OUTPUT_FOLDER, RUN_ID);

// Ensure run folder exists
if (!fs.existsSync(RUN_FOLDER)) {
  fs.mkdirSync(RUN_FOLDER, { recursive: true });
}

// Modify the MODEL_CONFIGS to include the number of workers
const MODEL_CONFIGS: (ModelConfig & { workers: number })[] = [
  {
    model: "HighQuality",
    ocrStrategy: "Auto",
    percentage: 60,
    workers: WORKERS_PER_CONFIG,
    segmentationStrategy: "LayoutAnalysis",
    testType: "standard",
  },
  {
    model: "HighQuality",
    ocrStrategy: "All",
    percentage: 30,
    workers: WORKERS_PER_CONFIG,
    segmentationStrategy: "LayoutAnalysis",
    testType: "standard",
  },
  {
    model: "HighQuality",
    ocrStrategy: "Auto",
    percentage: 5,
    workers: WORKERS_PER_CONFIG,
    segmentationStrategy: "LayoutAnalysis",
    testType: "structured",
  },
  {
    model: "HighQuality",
    ocrStrategy: "Auto",
    percentage: 5,
    workers: WORKERS_PER_CONFIG,
    segmentationStrategy: "Page",
    testType: "structured",
  },
];

// Save configurations to a txt file
const configFilePath = path.join(RUN_FOLDER, "config.txt");
const configData = {
  MODEL_CONFIGS,
  MAX_FILES_TO_PROCESS,
  REQUESTS_PER_SECOND: CONCURRENT_REQUESTS_PER_WORKER,
};
fs.writeFileSync(configFilePath, JSON.stringify(configData, null, 2));

function createCsvWriter(
  model: Model,
  ocrStrategy: OcrStrategy,
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
    console.log(`[DEBUG] Starting request for ${filePath} with config:`, {
      model: config.model,
      ocrStrategy: config.ocrStrategy,
      testType: config.testType,
      segmentationStrategy: config.segmentationStrategy,
    });

    const form = new FormData();

    // Add file with proper content type
    const fileBuffer = fs.readFileSync(filePath);
    form.append("file", fileBuffer, {
      filename: path.basename(filePath),
      contentType: "application/pdf",
    });

    // Add other fields
    form.append("model", config.model);
    form.append("target_chunk_length", "512");
    form.append("ocr_strategy", config.ocrStrategy);
    form.append("segmentation_strategy", config.segmentationStrategy);

    if (config.testType === "structured") {
      console.log(`[DEBUG] Adding JSON schema for structured extraction`);
      // Create the JSON schema with the correct format
      const schema = {
        title: "Document Metadata",
        type: "object",
        properties: [
          {
            name: "title",
            title: "Document Title",
            type: "string",
            description: "The main title of the document",
            default: null,
          },
          {
            name: "author",
            title: "Author",
            type: "string",
            description: "The author(s) of the document",
            default: null,
          },
          {
            name: "date_published",
            title: "Date Published",
            type: "string",
            description: "The publication date of the document",
            default: null,
          },
          {
            name: "location",
            title: "Location",
            type: "string",
            description: "The location mentioned in the document",
            default: null,
          },
        ],
      };

      // Append the JSON schema with the correct content type
      form.append("json_schema", JSON.stringify(schema), {
        contentType: "application/json",
      });
    }

    try {
      const response = await axios.post<TaskResponse>(API_URL, form, {
        headers: {
          ...form.getHeaders(),
          Authorization: API_KEY,
        },
      });
      return response.data;
    } catch (error: any) {
      if (error.response) {
        console.error("API Error Response:", {
          status: error.response.status,
          data: error.response.data,
        });
      }
      return null;
    }
  } catch (error) {
    console.error("Form creation failed:", error);
    return null;
  }
}

async function pollTask(
  taskId: string,
  config: ModelConfig
): Promise<TaskResponse | null> {
  console.log(`[DEBUG] Starting pollTask for ${taskId} with config:`, {
    model: config.model,
    ocrStrategy: config.ocrStrategy,
    testType: config.testType,
  });

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

      if (task.status === "Succeeded") {
        console.log(
          `[DEBUG] Task ${taskId} succeeded. Checking structured conditions:`,
          {
            hasOutput: !!task.output,
            isStructured: config.testType === "structured",
            output: task.output, // Log the actual output
          }
        );

        if (task.output && config.testType === "structured") {
          try {
            console.log(
              `[DEBUG] Processing structured output for task ${taskId}`
            );

            // Create a consistent folder name for structured outputs
            const folderName = `${config.model.toLowerCase()}_structured`;
            const configFolder = path.join(RUN_FOLDER, folderName);

            console.log(`[DEBUG] Attempting to create folder: ${configFolder}`);
            if (!fs.existsSync(configFolder)) {
              fs.mkdirSync(configFolder, { recursive: true });
              console.log(`[DEBUG] Created config folder: ${configFolder}`);
            }

            // Create outputs subfolder
            const outputsFolder = path.join(configFolder, "structured_outputs");
            console.log(
              `[DEBUG] Attempting to create outputs folder: ${outputsFolder}`
            );
            if (!fs.existsSync(outputsFolder)) {
              fs.mkdirSync(outputsFolder, { recursive: true });
              console.log(`[DEBUG] Created outputs folder: ${outputsFolder}`);
            }

            // Save structured output
            const outputFileName = `${task.file_name.replace(/\.[^/.]+$/, "")}_output.json`;
            const outputPath = path.join(outputsFolder, outputFileName);

            const outputData = {
              file_name: task.file_name,
              task_id: task.task_id,
              output: task.output,
            };

            fs.writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
            console.log(
              `[DEBUG] Successfully saved structured output to: ${outputPath}`
            );
          } catch (error) {
            console.error(
              `[DEBUG] Error saving structured output for task ${taskId}:`,
              error
            );
          }
        }
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

  // Create a copy of files array and shuffle it
  const shuffledFiles = [...files].sort(() => Math.random() - 0.5);
  let fileIndex = 0;

  configs.forEach((config) => {
    const configKey = `${config.model}_${config.ocrStrategy}`;
    const filesToProcess = Math.floor(
      (MAX_FILES_TO_PROCESS * config.percentage) / 100
    );
    distribution.set(configKey, []);

    for (
      let i = 0;
      i < filesToProcess && fileIndex < shuffledFiles.length;
      i++
    ) {
      distribution.get(configKey)!.push(shuffledFiles[fileIndex]);
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

  const limit = pLimit(CONCURRENT_REQUESTS_PER_WORKER);

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
  // Update file reading to include all files
  const files = fs.readdirSync(INPUT_FOLDER);

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
    const filesPerWorker = Math.ceil(assignedFiles.length / config.workers);

    for (let i = 0; i < config.workers; i++) {
      const workerFiles = assignedFiles.slice(
        i * filesPerWorker,
        (i + 1) * filesPerWorker
      );

      const worker = new Worker(__filename, {
        workerData: {
          config,
          assignedFiles: workerFiles,
          RUN_FOLDER,
          workerId: i + 1, // Add worker ID
        },
      });

      worker.on("message", (message: { type: string; data: WorkerResult }) => {
        if (message.type === "workerComplete") {
          workerResults.push(message.data);
          updateWorkerResultFile(config, message.data, i + 1); // Pass worker ID
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
  const {
    config,
    assignedFiles,
    RUN_FOLDER: workerRunFolder,
    workerId,
  } = workerData;
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
  result: WorkerResult & { failureTypes: FailureTypes },
  workerId: number
) {
  // Create a single results file for the configuration
  const resultFilePath = path.join(
    RUN_FOLDER,
    `${config.model}_${config.ocrStrategy}_results.txt`
  );

  const duration = (result.endTime - result.startTime) / 1000;
  const totalFailedFiles =
    result.failureTypes.startTaskFailed +
    result.failureTypes.pollTaskFailed +
    result.failureTypes.taskStatusFailed;

  // Append worker results to the file
  const workerContent = `
Worker ${workerId} Results:
Total pages processed: ${result.totalPages}
Failed to start task: ${result.failureTypes.startTaskFailed}
Failed to poll task: ${result.failureTypes.pollTaskFailed}
Tasks completed with failure status: ${result.failureTypes.taskStatusFailed}
Total failed files: ${totalFailedFiles}
Duration: ${duration.toFixed(2)} seconds
Pages per second: ${(result.totalPages / duration).toFixed(2)}
----------------------------------------
`;

  // Append the worker results to the file
  fs.appendFileSync(resultFilePath, workerContent);
}

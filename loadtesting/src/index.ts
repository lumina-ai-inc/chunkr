import { Worker, isMainThread, workerData } from "worker_threads";
import os from "os";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import dotenv from "dotenv";
import { createObjectCsvWriter } from "csv-writer";
import { TaskResponse } from "./models.js";
import pLimit from "p-limit";
import { fileURLToPath } from "url";
import path from "path";
import { EventEmitter } from "events";

dotenv.config();

const API_URL = "https://api-dev.chunkr.ai/api/v1/task";
const API_KEY = process.env.API_KEY;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FILE_PATH = path.join(__dirname, "..", "src", "test.pdf");

if (!API_KEY) {
  console.error("API_KEY not found in environment variables");
  process.exit(1);
}

const MAX_CONCURRENT_REQUESTS = 1000;
const limit = pLimit(MAX_CONCURRENT_REQUESTS);

const REQUESTS_PER_MODEL = 10;
const eventEmitter = new EventEmitter();

function createCsvWriter(model: "HighQuality" | "Fast") {
  const filePath = path.resolve(`task_results_${model.toLowerCase()}.csv`);

  if (!fs.existsSync(filePath)) {
    const headers =
      "task_id,file_name,page_count,message,start_time,end_time\n";
    fs.writeFileSync(filePath, headers);
  }

  return createObjectCsvWriter({
    path: filePath,
    header: [
      { id: "task_id", title: "task_id" },
      { id: "file_name", title: "file_name" },
      { id: "page_count", title: "page_count" },
      { id: "message", title: "message" },
      { id: "start_time", title: "start_time" },
      { id: "end_time", title: "end_time" },
    ],
    append: true,
  });
}

async function makeRequest(model: "HighQuality" | "Fast") {
  const form = new FormData();
  form.append("file", fs.createReadStream(FILE_PATH));
  form.append("model", model);
  form.append("target_chunk_length", "512");
  form.append("ocr_strategy", "All");

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
}

async function pollTask(taskId: string, model: "HighQuality" | "Fast") {
  console.log(`Starting task ${taskId} for model ${model}`);
  const csvWriter = createCsvWriter(model);
  let lastMessage = "";
  let messageStartTime = new Date().toISOString();

  while (true) {
    try {
      const response = await axios.get<TaskResponse>(`${API_URL}/${taskId}`, {
        headers: { Authorization: API_KEY },
      });
      const task = response.data;
      const currentTime = new Date().toISOString();

      if (task.message !== lastMessage) {
        if (lastMessage !== "") {
          await csvWriter.writeRecords([
            {
              task_id: task.task_id,
              file_name: task.file_name,
              page_count: task.page_count,
              message: lastMessage,
              start_time: messageStartTime,
              end_time: currentTime,
            },
          ]);
        }

        lastMessage = task.message;
        messageStartTime = currentTime;
      }

      if (task.status === "Succeeded" || task.status === "Failed") {
        await csvWriter.writeRecords([
          {
            task_id: task.task_id,
            file_name: task.file_name,
            page_count: task.page_count,
            message: lastMessage,
            start_time: messageStartTime,
            end_time: currentTime,
          },
        ]);
        console.log(`Task ${taskId} finished with status: ${task.status}`);
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    } catch (error) {
      break;
    }
  }
}

async function runLoadTest(model: "HighQuality" | "Fast") {
  const requestsPerSecond = 10;
  const interval = 1000 / requestsPerSecond;
  let requestCount = 0;

  while (requestCount < REQUESTS_PER_MODEL) {
    const startTime = Date.now();

    await limit(async () => {
      const task = await makeRequest(model);
      if (task) {
        pollTask(task.task_id, model).catch(console.error);
        requestCount++;

        if (requestCount === REQUESTS_PER_MODEL) {
          eventEmitter.emit(`${model}Complete`);
        }
      }
    });

    const elapsedTime = Date.now() - startTime;
    const delay = Math.max(0, interval - elapsedTime);
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
}

if (isMainThread) {
  const numWorkers = 4;
  console.log(`Starting ${numWorkers} workers...`);
  let completedModels = 0;

  for (let i = 0; i < numWorkers; i++) {
    const model = i % 2 === 0 ? "HighQuality" : "Fast";
    const worker = new Worker(__filename, {
      workerData: { model },
    });

    worker.on("error", (error) => {
      console.error(`Worker error: ${error}`);
    });

    worker.on("exit", (code) => {
      if (code !== 0) {
        console.error(`Worker stopped with exit code ${code}`);
      }
    });

    eventEmitter.on(`${model}Complete`, () => {
      completedModels++;
      if (completedModels === 2) {
        console.log("Load test completed for both models. Exiting...");
        process.exit(0);
      }
    });
  }
} else {
  // This code runs in worker threads
  runLoadTest(workerData.model);
}

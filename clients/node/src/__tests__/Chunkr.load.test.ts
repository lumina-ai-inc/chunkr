//LOAD TEST INSTRUCTIONS:
//1. Create input directory in __tests__ with an appropriate number of files
//2. Set MAX_FILES in the environment to the number of files you want to process
//3. Run the test: Open node-client in terminal and run "npm run prepare && npm run test:load"
//4. Check the logs directory for the results - saved in logs/processing-log-<timestamp>.json
//5. Check the output directory for the outputs - saved in output/<timestamp>/<filename>.result.json
//6. TEST TIMEOUT IS 1 HOUR - if you want to run longer, increase the timeout in the jest.load.config.js file

import { Chunkr } from "../Chunkr";
import { Status } from "../models/TaskResponseData";
import { describe, it, expect, beforeAll, afterAll } from "@jest/globals";
import * as fs from "fs/promises";
import * as path from "path";

const INPUT_DIR = path.join(__dirname, "input");
const OUTPUT_DIR = path.join(__dirname, "output");
const LOGS_DIR = path.join(__dirname, "logs");
const POLL_INTERVAL = 1000; // 5 seconds

interface ProcessingStats {
  fileName: string;
  taskId: string;
  status: Status;
  startTime: number;
  endTime?: number;
  pageCount?: number | null;
  error?: string;
}

const LOG_WRITE_INTERVAL = 2000;

const logQueue = {
  entries: [] as ProcessingStats[],
};

let lastLogWrite = 0;
let currentLogPath: string;

async function writeQueuedLogs(force: boolean = false) {
  const now = Date.now();
  if (!force && now - lastLogWrite < LOG_WRITE_INTERVAL) {
    return;
  }

  try {
    await fs.mkdir(LOGS_DIR, { recursive: true });
    const logPath = currentLogPath;

    // Skip if no entries yet
    if (logQueue.entries.length === 0) {
      return;
    }

    const totalPages = logQueue.entries.reduce(
      (sum, stat) => sum + (stat.pageCount || 0),
      0,
    );
    const successCount = logQueue.entries.filter(
      (s) => s.status === Status.SUCCEEDED,
    ).length;
    const failureCount = logQueue.entries.filter(
      (s) => s.status === Status.FAILED,
    ).length;
    const inProgressCount = logQueue.entries.filter(
      (s) => ![Status.SUCCEEDED, Status.FAILED].includes(s.status),
    ).length;

    // Find the earliest start time from all entries
    const firstStartTime = Math.min(
      ...logQueue.entries.map((entry) => entry.startTime),
    );

    const summary = {
      totalFiles: logQueue.entries.length,
      successCount,
      failureCount,
      inProgressCount,
      totalPages,
      totalTimeSeconds: (now - firstStartTime) / 1000,
      pagesPerSecond: totalPages / ((now - firstStartTime) / 1000),
      details: logQueue.entries,
      lastUpdated: new Date().toISOString(),
    };

    await fs.writeFile(logPath, JSON.stringify(summary, null, 2));
    lastLogWrite = now;

    // Log progress to console
    console.log(
      `Progress: ${successCount} succeeded, ${failureCount} failed, ${inProgressCount} in progress`,
    );
  } catch (error) {
    console.error("Error writing log file:", error);
  }
}

// Add batch size constant for polling
const POLL_BATCH_SIZE = 10; // Poll 10 tasks at a time

describe("Chunkr Load Test", () => {
  let chunkr: Chunkr;
  let logInterval: NodeJS.Timeout;
  let currentOutputDir: string;

  beforeAll(async () => {
    chunkr = new Chunkr();
    if (!process.env.CHUNKR_API_KEY) {
      throw new Error("CHUNKR_API_KEY not found in environment");
    }

    // Check if input directory exists
    try {
      await fs.access(INPUT_DIR);
    } catch (error) {
      throw new Error(
        `Input directory not found at ${INPUT_DIR}. Please create the directory and add test files before running the load test.`,
      );
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    currentOutputDir = path.join(OUTPUT_DIR, timestamp);
    await fs.mkdir(currentOutputDir, { recursive: true });
    currentLogPath = path.join(LOGS_DIR, `processing-log-${timestamp}.json`);
  });

  afterAll(async () => {
    // Wait for any pending log writes and cleanup
    if (logInterval) {
      clearInterval(logInterval);
    }
    // Add delay to ensure final writes complete
    await new Promise((resolve) => setTimeout(resolve, LOG_WRITE_INTERVAL));
    await writeQueuedLogs(true);
  });

  it("should process files and track completion status", async () => {
    // Set up interval for log writing
    logInterval = setInterval(() => writeQueuedLogs(), LOG_WRITE_INTERVAL);

    const startTime = Date.now();
    const maxFiles = process.env.MAX_FILES
      ? parseInt(process.env.MAX_FILES)
      : undefined;
    let fileNames = await fs.readdir(INPUT_DIR);
    // Shuffle the files array
    fileNames = fileNames.sort(() => Math.random() - 0.5);

    if (maxFiles) {
      fileNames = fileNames.slice(0, maxFiles);
      console.log(
        `Processing ${fileNames.length} randomly selected files (limited by MAX_FILES=${maxFiles})`,
      );
    } else {
      console.log(`Processing all ${fileNames.length} files in random order`);
    }

    // Initial upload of all files
    for (const file of fileNames) {
      const inputPath = path.join(INPUT_DIR, file);
      console.log(`Uploading file: ${file}`);
      try {
        const result = await chunkr.createTask(inputPath);
        console.log(`Successfully created task ${result.task_id} for ${file}`);
        const stats: ProcessingStats = {
          fileName: result.output?.file_name || file,
          taskId: result.task_id,
          status: result.status,
          startTime: Date.now(),
        };
        logQueue.entries.push(stats);
        await writeQueuedLogs();
      } catch (error: any) {
        console.error(`Failed to upload ${file}:`, error);
        const stats: ProcessingStats = {
          fileName: file,
          taskId: "UPLOAD_FAILED",
          status: Status.FAILED,
          startTime: Date.now(),
          error: error?.message || "Unknown error",
        };
        logQueue.entries.push(stats);
        await writeQueuedLogs();
      }
    }

    // Poll for completion
    let allCompleted = false;
    let pendingWrites: Promise<void>[] = [];

    while (!allCompleted) {
      allCompleted = true;
      let updatedEntries = false;

      // Get all pending tasks (not succeeded or failed)
      const pendingTasks = logQueue.entries.filter(
        (stat) => ![Status.SUCCEEDED, Status.FAILED].includes(stat.status),
      );

      // Process tasks in batches
      for (let i = 0; i < pendingTasks.length; i += POLL_BATCH_SIZE) {
        const batch = pendingTasks.slice(i, i + POLL_BATCH_SIZE);

        // Poll batch concurrently
        const pollResults = await Promise.allSettled(
          batch.map((stat) => chunkr.getTask(stat.taskId)),
        );

        // Process results
        pollResults.forEach((result, index) => {
          const stat = batch[index];

          if (result.status === "fulfilled") {
            const taskResponse = result.value;

            // Only update if status changed
            if (taskResponse.status !== stat.status) {
              stat.status = taskResponse.status;
              updatedEntries = true;

              if (taskResponse.status === Status.SUCCEEDED) {
                stat.endTime = Date.now();
                stat.pageCount = taskResponse.output?.page_count;

                // Queue file write instead of immediate write
                const outputPath = path.join(
                  currentOutputDir,
                  `${stat.fileName}.result.json`,
                );
                pendingWrites.push(
                  fs.writeFile(
                    outputPath,
                    JSON.stringify(taskResponse, null, 2),
                  ),
                );
              } else if (taskResponse.status === Status.FAILED) {
                stat.endTime = Date.now();
                stat.error = taskResponse.error || "Unknown error";
              }
            }
          } else {
            console.error(`Error polling task ${stat.taskId}:`, result.reason);
          }
        });

        // Check if any tasks are still pending
        if (
          pendingTasks.some(
            (stat) => ![Status.SUCCEEDED, Status.FAILED].includes(stat.status),
          )
        ) {
          allCompleted = false;
        }

        // Force log update if entries changed
        if (updatedEntries) {
          await writeQueuedLogs(true);
        }
      }

      if (!allCompleted) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
      }
    }

    // Ensure final operations complete before test ends
    clearInterval(logInterval);
    await writeQueuedLogs(true);

    // Add small delay to ensure all async operations complete
    await new Promise((resolve) => setTimeout(resolve, LOG_WRITE_INTERVAL));

    // Verify all files were processed
    const succeededCount = logQueue.entries.filter(
      (s) => s.status === Status.SUCCEEDED,
    ).length;
    expect(succeededCount).toBeGreaterThan(0);
  }, 3600000); // 1 hour timeout
});

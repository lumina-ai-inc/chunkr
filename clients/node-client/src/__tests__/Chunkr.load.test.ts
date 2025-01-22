//LOAD TEST INSTRUCTIONS:
//1. Create input directory in __tests__ with an appropriate number of files
//2. Set MAX_FILES in the environment to the number of files you want to process
//3. Run the test: Open node-client in terminal and run "npm run prepare && npm run test:load"
//4. Check the logs directory for the results - saved in logs/processing-log-<timestamp>.json
//5. Check the output directory for the outputs - saved in output/<timestamp>/<filename>.result.json
//6. TEST TIMEOUT IS 1 HOUR

import { Chunkr } from "../Chunkr";
import { Status } from "../models/TaskResponseData";
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
  lastLogWrite = now;

  if (logQueue.entries.length === 0) {
    return;
  }

  await fs.mkdir(LOGS_DIR, { recursive: true });
  const logPath = currentLogPath;

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

  const summary = {
    totalFiles: logQueue.entries.length,
    successCount,
    failureCount,
    totalPages,
    totalTimeSeconds: (now - logQueue.entries[0].startTime) / 1000,
    pagesPerSecond: totalPages / ((now - logQueue.entries[0].startTime) / 1000),
    details: logQueue.entries,
    lastUpdated: new Date().toISOString(),
  };

  await fs.writeFile(logPath, JSON.stringify(summary, null, 2));
  console.log(`Log file updated: ${logPath}`);
}

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
    let files = await fs.readdir(INPUT_DIR);

    // Shuffle the files array
    files = files.sort(() => Math.random() - 0.5);

    if (maxFiles) {
      files = files.slice(0, maxFiles);
      console.log(
        `Processing ${files.length} randomly selected files (limited by MAX_FILES=${maxFiles})`,
      );
    } else {
      console.log(`Processing all ${files.length} files in random order`);
    }

    // Initial upload of all files
    for (const file of files) {
      const inputPath = path.join(INPUT_DIR, file);
      const content = await fs.readFile(inputPath);

      try {
        const result = await chunkr.createTask(content);
        const stats: ProcessingStats = {
          fileName: result.file_name || file,
          taskId: result.task_id,
          status: result.status,
          startTime: Date.now(),
        };
        logQueue.entries.push(stats);
        await writeQueuedLogs();
      } catch (error: any) {
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

      for (const stat of logQueue.entries) {
        if (stat.status !== Status.SUCCEEDED && stat.status !== Status.FAILED) {
          try {
            const taskResponse = await chunkr.getTask(stat.taskId);
            stat.status = taskResponse.status;

            if (taskResponse.status === Status.SUCCEEDED) {
              stat.endTime = Date.now();
              stat.pageCount = taskResponse.page_count;

              // Save the complete TaskResponse object
              const outputPath = path.join(
                currentOutputDir,
                `${stat.fileName}.result.json`,
              );
              pendingWrites.push(
                fs.writeFile(outputPath, JSON.stringify(taskResponse, null, 2)),
              );
            } else if (taskResponse.status === Status.FAILED) {
              stat.endTime = Date.now();
              stat.error = taskResponse.error || "Unknown error";
            } else {
              allCompleted = false;
            }
          } catch (error) {
            console.error(`Error polling task ${stat.taskId}:`, error);
            allCompleted = false;
          }
        }
      }

      // Wait for all pending file writes to complete
      if (pendingWrites.length > 0) {
        await Promise.all(pendingWrites);
        pendingWrites = [];
      }

      // Make writeQueuedLogs awaited
      await writeQueuedLogs();

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

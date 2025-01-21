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

async function writeLog(stats: ProcessingStats[], totalTime: number) {
  await fs.mkdir(LOGS_DIR, { recursive: true });

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logPath = path.join(LOGS_DIR, `processing-log-${timestamp}.json`);

  const totalPages = stats.reduce(
    (sum, stat) => sum + (stat.pageCount || 0),
    0,
  );
  const successCount = stats.filter(
    (s) => s.status === Status.SUCCEEDED,
  ).length;
  const failureCount = stats.filter((s) => s.status === Status.FAILED).length;

  const summary = {
    totalFiles: stats.length,
    successCount,
    failureCount,
    totalPages,
    totalTimeSeconds: totalTime / 1000,
    pagesPerSecond: totalPages / (totalTime / 1000),
    details: stats,
  };

  await fs.writeFile(logPath, JSON.stringify(summary, null, 2));
  return logPath;
}

describe("Chunkr Load Test", () => {
  let chunkr: Chunkr;

  beforeAll(async () => {
    // Ensure we have API key
    if (!process.env.CHUNKR_API_KEY) {
      throw new Error("CHUNKR_API_KEY not found in environment");
    }

    // Create output directory if it doesn't exist
    await fs.mkdir(OUTPUT_DIR, { recursive: true });

    chunkr = new Chunkr();
  });

  it("should process files and track completion status", async () => {
    const startTime = Date.now();
    const files = await fs.readdir(INPUT_DIR);
    const processingStats: ProcessingStats[] = [];

    // Initial upload of all files
    for (const file of files) {
      const inputPath = path.join(INPUT_DIR, file);
      const content = await fs.readFile(inputPath);

      try {
        const result = await chunkr.createTask(content);
        processingStats.push({
          fileName: result.file_name || file,
          taskId: result.task_id,
          status: result.status,
          startTime: Date.now(),
        });
      } catch (error: any) {
        processingStats.push({
          fileName: file,
          taskId: "UPLOAD_FAILED",
          status: Status.FAILED,
          startTime: Date.now(),
          error: error?.message || "Unknown error",
        });
      }
    }

    // Poll for completion
    let allCompleted = false;
    while (!allCompleted) {
      allCompleted = true;

      for (const stat of processingStats) {
        if (stat.status !== Status.SUCCEEDED && stat.status !== Status.FAILED) {
          try {
            const taskResponse = await chunkr.getTask(stat.taskId);
            stat.status = taskResponse.status;

            if (taskResponse.status === Status.SUCCEEDED) {
              stat.endTime = Date.now();
              stat.pageCount = taskResponse.page_count;

              // Save the complete TaskResponse object
              const outputPath = path.join(
                OUTPUT_DIR,
                `${stat.fileName}.result.json`,
              );
              await fs.writeFile(
                outputPath,
                JSON.stringify(taskResponse, null, 2),
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

      if (!allCompleted) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
      }
    }

    const totalTime = Date.now() - startTime;
    const logPath = await writeLog(processingStats, totalTime);

    console.log(`Processing complete. Log file written to: ${logPath}`);

    // Verify all files were processed
    const succeededCount = processingStats.filter(
      (s) => s.status === Status.SUCCEEDED,
    ).length;
    expect(succeededCount).toBeGreaterThan(0);
  }, 300000); // 5 minute timeout for longer runs
});

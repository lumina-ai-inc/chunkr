import { runBatch } from "./batch/batchProcessor.js";
import { startLoadTest } from "./loadTest/loadTester.js";
import { TaskConfig } from "./models.js";

const isLoadTest = process.env.LOAD_TEST === "true";

async function main() {
  if (isLoadTest) {
    console.log("Starting load test from index.ts...");
    await startLoadTest();
  } else {
    console.log("Running single batch...");
    const config: TaskConfig = {
      model: "HighQuality",
      ocr_strategy: "Auto",
      target_chunk_length: 512,
    };
    const filePaths = ["path/to/your/folder"]; // Replace with actual folder path
    await runBatch(config, filePaths);
  }
}

main().catch((error) => {
  console.error("Error in main:", error);
  process.exit(1);
});

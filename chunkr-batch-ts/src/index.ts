import { runBatch } from "./batch/batchProcessor.js";
import { startLoadTest } from "./loadTest/loadTester.js";
import { INPUT_FOLDER } from "./config.js";
import fs from "fs/promises";
import path from "path";
import { TaskConfig } from "./models";

async function main() {
  const files = await fs.readdir(INPUT_FOLDER);
  const filePaths = files.map((file) => path.join(INPUT_FOLDER, file));

  if (process.env.LOAD_TEST === "true") {
    await startLoadTest();
  } else {
    // Regular batch processing
    const config: TaskConfig = {
      model: "HighQuality",
      ocr_strategy: "Auto",
      target_chunk_length: 512,
    };
    await runBatch(config, filePaths);

    config.model = "Fast";
    await runBatch(config, filePaths);
  }
}

main().catch(console.error);

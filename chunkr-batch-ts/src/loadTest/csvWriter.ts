import { createObjectCsvWriter } from "csv-writer";
import path from "path";
import { existsSync, mkdirSync, writeFile } from "fs";
import { ModelOCRStrategy } from "../models.js";

const OUTPUT_FOLDER = path.resolve("output");

export function createCsvWriter(modelOCRStrategy: ModelOCRStrategy) {
  if (!existsSync(OUTPUT_FOLDER)) {
    mkdirSync(OUTPUT_FOLDER);
  }

  const filePath = path.join(
    OUTPUT_FOLDER,
    `load_test_results_${modelOCRStrategy.toLowerCase()}.csv`
  );

  if (!existsSync(filePath)) {
    const headers =
      "file_name,start_time,end_time,model,ocr_strategy,target_chunk_length,task_id,status,message\n";
    writeFile(filePath, headers, (err) => {
      if (err) console.error("Error writing CSV header:", err);
    });
  }

  return createObjectCsvWriter({
    path: filePath,
    header: [
      { id: "file_name", title: "file_name" },
      { id: "start_time", title: "start_time" },
      { id: "end_time", title: "end_time" },
      { id: "model", title: "model" },
      { id: "ocr_strategy", title: "ocr_strategy" },
      { id: "target_chunk_length", title: "target_chunk_length" },
      { id: "task_id", title: "task_id" },
      { id: "status", title: "status" },
      { id: "message", title: "message" },
    ],
    append: true,
  });
}

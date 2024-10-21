import { createObjectCsvWriter } from "csv-writer";
import path from "path";
import { existsSync, writeFile } from "fs";
import { ModelOCRStrategy } from "../models.js";

export function createCsvWriter(modelOCRStrategy: ModelOCRStrategy) {
  const filePath = path.resolve(
    `load_test_results_${modelOCRStrategy.toLowerCase()}.csv`
  );

  if (!existsSync(filePath)) {
    const headers =
      "file_name,start_time,end_time,model,ocr_strategy,target_chunk_length\n";
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
    ],
    append: true,
  });
}

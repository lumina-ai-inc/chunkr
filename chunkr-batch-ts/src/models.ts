export type Model = "Fast" | "HighQuality";
export type OCRStrategy = "Auto" | "All" | "Off";
export type ModelOCRStrategy = `${Model}_${OCRStrategy}`;

export interface TaskConfig {
  model: Model;
  ocr_strategy: OCRStrategy;
  target_chunk_length: number;
}

export interface TaskResponse {
  task_id: string;
  status: "Starting" | "Processing" | "Succeeded" | "Failed" | "Cancelled";
  file_name: string;
  page_count: number;
  message: string;
}

export interface TaskResponse {
  configuration: {
    model: "HighQuality" | "Fast";
    ocr_strategy: string;
    target_chunk_length: number;
  };
  created_at: string;
  expires_at?: string;
  file_name: string;
  finished_at?: string;
  input_file_url: string;
  message: string;
  output?: any[];
  page_count: number;
  status: "Starting" | "Processing" | "Succeeded" | "Failed" | "Cancelled";
  task_id: string;
  task_url: string;
}

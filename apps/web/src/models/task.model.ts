import { Model } from "./upload.model"; // Assuming Model is defined in a separate file
import { Chunk } from "./chunk.model";

export enum Status {
  Starting = "Starting",
  Processing = "Processing",
  Succeeded = "Succeeded",
  Failed = "Failed",
  Canceled = "Canceled",
}

export interface Output {
  chunks: Chunk[];
  extracted_json: ExtractedJson;
}

export interface TaskResponse {
  task_id: string;
  status: Status;
  created_at: Date;
  finished_at: string | null;
  expires_at: Date | null;
  message: string;
  input_file_url: string | null;
  pdf_url: string | null;
  // eslint-disable-next-line
  output: any;
  task_url: string | null;
  configuration: Configuration;
  file_name: string | null;
  page_count: number | null;
}

export interface Configuration {
  model: Model;
  ocr_strategy: "All" | "Auto" | "Off";
  target_chunk_length: number | null;
}

export interface ExtractedJson {
  title: string;
  schema_type: string;
  extracted_fields: ExtractedField[];
}

// Define a JSON value type that matches serde_json::Value capabilities
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface ExtractedField {
  name: string;
  field_type: string;
  value: JsonValue;
}

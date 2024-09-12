import { Model } from "./upload.model"; // Assuming Model is defined in a separate file

export enum Status {
  Starting = "Starting",
  Processing = "Processing",
  Succeeded = "Succeeded",
  Failed = "Failed",
  Canceled = "Canceled",
}

export interface TaskResponse {
  task_id: string;
  status: Status;
  created_at: Date;
  finished_at: string | null;
  expires_at: Date | null;
  message: string;
  input_file_url: string | null;
  // eslint-disable-next-line 
  output: any;
  task_url: string | null;
  configuration: Configuration;
}

export interface Configuration {
  model: Model;
  target_chunk_length: number | null;
}

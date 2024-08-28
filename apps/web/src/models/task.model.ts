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
  expiration_time: Date | null;
  message: string;
  input_file_url: string | null;
  output_file_url: string | null;
  task_url: string | null;
  model: Model;
}

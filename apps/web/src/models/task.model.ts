import { Model } from "./upload.model"; // Assuming Model is defined in a separate file

export enum Status {
  Starting = "Starting",
  Processing = "Processing",
  Succeeded = "Succeeded",
  Failed = "Failed",
  Canceled = "Canceled",
}

export interface TaskResponse {
  taskId: string;
  status: Status;
  createdAt: Date;
  finishedAt: string | null;
  expirationTime: Date | null;
  message: string;
  fileUrl: string | null;
  taskUrl: string | null;
  model: Model;
}

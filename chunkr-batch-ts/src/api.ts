import axios from "axios";
import FormData from "form-data";
import { createReadStream } from "fs";
import { TaskResponse, TaskConfig } from "./models.js";

const axiosInstance = axios.create({
  baseURL: process.env.API_BASE_URL,
  headers: {
    Authorization: process.env.API_KEY,
  },
});

export async function makeRequest(
  config: TaskConfig,
  filePath: string
): Promise<TaskResponse | null> {
  const form = new FormData();
  form.append("file", createReadStream(filePath));
  form.append("model", config.model);
  form.append("target_chunk_length", config.target_chunk_length.toString());
  form.append("ocr_strategy", config.ocr_strategy);

  try {
    const response = await axiosInstance.post<TaskResponse>("", form, {
      headers: {
        ...form.getHeaders(),
      },
    });
    return response.data;
  } catch (error) {
    console.error(`Error processing file ${filePath}:`, error);
    return null;
  }
}

export async function pollTask(taskId: string): Promise<TaskResponse> {
  const response = await axiosInstance.get<TaskResponse>(`/${taskId}`);
  return response.data;
}

import axios, { AxiosInstance } from "axios";
import { Configuration } from "./models/Configuration";
import { TaskResponse } from "./models/TaskResponse";
import { HeadersMixin } from "./auth/HeadersMixin";
import { prepareUploadData, FileInput } from "./utils/prepareUploadData";
import * as dotenv from "dotenv";

export class Chunkr extends HeadersMixin {
  private client: AxiosInstance;
  private url: string;

  constructor(apiKey?: string, url?: string) {
    super();
    dotenv.config();

    this.url = (url ||
      process.env.CHUNKR_URL ||
      "https://api.chunkr.ai") as string;
    this._apiKey = (apiKey || process.env.CHUNKR_API_KEY) as string;

    if (!this._apiKey) {
      throw new Error(
        "API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. You can get an api key at: https://www.chunkr.ai",
      );
    }

    this.url = this.url.replace(/\/$/, "");
    this.client = axios.create({
      baseURL: this.url,
      headers: this.headers(),
    });
  }

  async upload(file: FileInput, config?: Configuration): Promise<TaskResponse> {
    const task = await this.createTask(file, config);
    return task.poll();
  }

  async createTask(
    file: FileInput,
    config?: Configuration,
  ): Promise<TaskResponse> {
    const formData = await prepareUploadData(file, config);
    const response = await this.client.post("/api/v1/task", formData);
    return new TaskResponse(response.data, this);
  }

  async getTask(taskId: string): Promise<TaskResponse> {
    const response = await this.client.get(`/api/v1/task/${taskId}`);
    return new TaskResponse(response.data, this);
  }

  async updateTask(
    taskId: string,
    config: Configuration,
  ): Promise<TaskResponse> {
    const formData = await prepareUploadData(null, config);
    const response = await this.client.patch(
      `/api/v1/task/${taskId}`,
      formData,
    );
    return new TaskResponse(response.data, this);
  }

  async deleteTask(taskId: string): Promise<void> {
    await this.client.delete(`/api/v1/task/${taskId}`);
  }

  async cancelTask(taskId: string): Promise<void> {
    await this.client.get(`/api/v1/task/${taskId}/cancel`);
  }
}

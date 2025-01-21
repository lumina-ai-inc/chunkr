import axios, { AxiosInstance } from "axios";
import { Configuration } from "./models/Configuration";
import { TaskResponse } from "./models/TaskResponse";
import { ClientConfig } from "./models/ClientConfig";
import { prepareUploadData, FileInput } from "./utils/prepareUploadData";
import * as dotenv from "dotenv";
import type { TaskResponseData } from "./models/TaskResponseData";

export class Chunkr {
  private client: AxiosInstance;
  private config: ClientConfig;

  /**
   * Initialize a new Chunkr API client
   * @param {ClientConfig | string} [configOrApiKey] - Either a configuration object or API key string.
   *        If omitted, will use CHUNKR_API_KEY from environment.
   * @param {string} [url] - Optional API URL override.
   *        If omitted, will use CHUNKR_URL from environment, or default to "https://api.chunkr.ai"
   * @throws {Error} If API key is not provided via parameters or CHUNKR_API_KEY environment variable
   *
   * @example
   * ```typescript
   * // Using environment variables (CHUNKR_API_KEY and optionally CHUNKR_URL)
   * const client = new Chunkr();
   *
   * // Using direct API key
   * const client = new Chunkr("your-api-key");
   *
   * // Using configuration object
   * const client = new Chunkr({
   *   apiKey: "your-api-key",
   *   baseUrl: "https://custom-url.example.com"
   * });
   * ```
   */
  constructor(configOrApiKey?: ClientConfig | string, url?: string) {
    dotenv.config();

    // Handle different constructor signatures
    if (typeof configOrApiKey === "object") {
      this.config = {
        apiKey: configOrApiKey.apiKey,
        baseUrl:
          configOrApiKey.baseUrl ||
          process.env.CHUNKR_URL ||
          "https://api.chunkr.ai",
      };
    } else {
      this.config = {
        apiKey: (configOrApiKey || process.env.CHUNKR_API_KEY) as string,
        baseUrl: (url ||
          process.env.CHUNKR_URL ||
          "https://api.chunkr.ai") as string,
      };
    }

    if (!this.config.apiKey) {
      throw new Error(
        "API key must be provided either directly, in .env file, or as CHUNKR_API_KEY environment variable. You can get an api key at: https://www.chunkr.ai",
      );
    }

    this.config.baseUrl = this.config.baseUrl?.replace(/\/$/, "") || "";
    this.client = axios.create({
      baseURL: this.config.baseUrl,
      headers: this.getHeaders(),
    });
  }

  private getHeaders(): Record<string, string> {
    return {
      Authorization: this.config.apiKey,
    };
  }

  /**
   * Upload a file and wait for processing to complete.
   * @param {FileInput} file - The file to upload (can be a path string, Buffer, or ReadStream)
   * @param {Configuration} [config] - Optional configuration options for processing
   * @returns {Promise<TaskResponse>} The completed task response
   *
   * @example
   * ```typescript
   * // Upload from file path
   * const task = await chunkr.upload("document.pdf");
   *
   * // Upload with configuration
   * const task = await chunkr.upload("document.pdf", {
   *   ocr_strategy: OcrStrategy.AUTO,
   *   high_resolution: true
   * });
   * ```
   */
  async upload(file: FileInput, config?: Configuration): Promise<TaskResponse> {
    const task = await this.createTask(file, config);
    return task.poll();
  }

  /**
   * Create a new task and immediately return the task response.
   * It will not wait for processing to complete. To wait for the full processing to complete, use task.poll().
   * @param {FileInput} file - The file to upload (can be a path string, Buffer, or ReadStream)
   * @param {Configuration} [config] - Optional configuration options for processing
   * @returns {Promise<TaskResponse>} The initial task response
   *
   * @example
   * ```typescript
   * // Create task without waiting
   * const task = await chunkr.createTask("document.pdf");
   * // ... do other things ...
   * await task.poll(); // Wait for completion when needed
   * ```
   */
  async createTask(
    file: FileInput,
    config?: Configuration,
  ): Promise<TaskResponse> {
    const formData = await prepareUploadData(file, config);
    const response = await this.client.post<TaskResponseData>(
      "/api/v1/task",
      formData,
    );
    return new TaskResponse(response.data, this);
  }

  /**
   * Get a task by its ID.
   * @param {string} taskId - The ID of the task to retrieve
   * @returns {Promise<TaskResponse>} The task response
   */
  async getTask(taskId: string): Promise<TaskResponse> {
    const response = await this.client.get(`/api/v1/task/${taskId}`);
    return new TaskResponse(response.data, this);
  }

  /**
   * Update a task by its ID and immediately return the task response.
   * It will not wait for processing to complete. To wait for the full processing to complete, use task.poll().
   * @param {string} taskId - The ID of the task to update
   * @param {Configuration} config - Configuration options for processing
   * @returns {Promise<TaskResponse>} The updated task response
   *
   * @example
   * ```typescript
   * // Update task configuration
   * const task = await chunkr.updateTask(taskId, {
   *   ocr_strategy: OcrStrategy.ALL
   * });
   * await task.poll(); // Wait for the update to complete
   * ```
   */
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

  /**
   * Delete a task by its ID.
   * @param {string} taskId - The ID of the task to delete
   * @returns {Promise<void>}
   * @throws {Error} If the task is currently processing
   */
  async deleteTask(taskId: string): Promise<void> {
    await this.client.delete(`/api/v1/task/${taskId}`);
  }

  /**
   * Cancel a task by its ID.
   * @param {string} taskId - The ID of the task to cancel
   * @returns {Promise<void>}
   * @throws {Error} If the task has already started processing
   */
  async cancelTask(taskId: string): Promise<void> {
    await this.client.get(`/api/v1/task/${taskId}/cancel`);
  }
}

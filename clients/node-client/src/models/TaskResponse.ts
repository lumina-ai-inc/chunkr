import { Chunkr } from "../Chunkr";
import { Configuration } from "./Configuration";
import { TaskResponseData, Output, Status } from "./TaskResponseData";

/**
 * Represents a response from a Chunkr API task operation.
 * Contains methods for polling, updating, and managing task state.
 */
export class TaskResponse implements TaskResponseData {
  #chunkr: Chunkr;

  public task_id!: string;
  public status!: Status;
  public created_at!: string;
  public finished_at!: string | null;
  public expires_at!: string | null;
  public message!: string;
  public input_file_url!: string | null;
  public pdf_url!: string | null;
  public output!: Output | null;
  public task_url!: string | null;
  public configuration!: Configuration;
  public file_name!: string | null;
  public page_count!: number | null;
  public error?: string;

  constructor(data: TaskResponseData, chunkr: Chunkr) {
    this.#chunkr = chunkr;
    Object.assign(this, data);
  }

  toJSON() {
    return {
      task_id: this.task_id,
      status: this.status,
      created_at: this.created_at,
      finished_at: this.finished_at,
      expires_at: this.expires_at,
      message: this.message,
      input_file_url: this.input_file_url,
      pdf_url: this.pdf_url,
      output: this.output,
      task_url: this.task_url,
      file_name: this.file_name,
      page_count: this.page_count,
      error: this.error,
    };
  }

  /**
   * Poll the task until it reaches a terminal state (Succeeded, Failed, or Cancelled).
   * @param {number} [interval=1000] - Polling interval in milliseconds
   * @returns {Promise<TaskResponse>} The completed task response
   * @throws {Error} If the task fails or reaches an unexpected state
   */
  async poll(interval: number = 1000): Promise<TaskResponse> {
    const pollingStates = [Status.STARTING, Status.PROCESSING];
    const terminalStates = [Status.SUCCEEDED, Status.FAILED, Status.CANCELLED];

    while (pollingStates.includes(this.status)) {
      try {
        await new Promise((resolve) => setTimeout(resolve, interval));
        const response = await this.#chunkr.getTask(this.task_id);
        Object.assign(this, response);
        console.debug(`Task ${this.task_id} status: ${this.status}`);
      } catch (error) {
        console.warn(`Polling error for task ${this.task_id}:`, error);
        continue;
      }
    }

    if (!terminalStates.includes(this.status)) {
      throw new Error(`Task ended in unexpected state: ${this.status}`);
    }

    if (this.status === Status.FAILED) {
      throw new Error(
        this.error || "Task failed without specific error message",
      );
    }

    return this;
  }

  /**
   * Cancel the current task. Only works if the task hasn't started processing.
   * @returns {Promise<void>}
   * @throws {Error} If the task has already started processing
   */
  async cancel(): Promise<void> {
    await this.#chunkr.cancelTask(this.task_id);
    await this.poll();
  }

  /**
   * Delete the current task.
   * @returns {Promise<void>}
   * @throws {Error} If the task is currently processing
   */
  async delete(): Promise<void> {
    await this.#chunkr.deleteTask(this.task_id);
  }

  /**
   * Get content from the task's output chunks.
   * @param {"html" | "markdown" | "content"} type - The type of content to retrieve
   * @returns {string} The concatenated content of all chunks
   */
  getContent(type: "html" | "markdown" | "content"): string {
    if (!this.output?.chunks) {
      return "";
    }

    return this.output.chunks
      .flatMap((chunk) =>
        chunk.segments.map((segment) => segment[type]).filter(Boolean),
      )
      .join("\n");
  }

  /**
   * Get HTML content from the task's output chunks.
   * @returns {string} The concatenated HTML content of all chunks
   */
  getHtml(): string {
    return this.getContent("html");
  }

  /**
   * Get Markdown content from the task's output chunks.
   * @returns {string} The concatenated Markdown content of all chunks
   */
  getMarkdown(): string {
    return this.getContent("markdown");
  }
}

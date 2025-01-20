import { Chunkr } from "../Chunkr";
import { Status } from "./Configuration";
import { TaskResult, Output } from "./TaskResult";

export class TaskResponse implements TaskResult {
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
  public file_name!: string | null;
  public page_count!: number | null;
  public error?: string;

  constructor(data: TaskResult, chunkr: Chunkr) {
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

  async cancel(): Promise<void> {
    await this.#chunkr.cancelTask(this.task_id);
    await this.poll();
  }

  async delete(): Promise<void> {
    await this.#chunkr.deleteTask(this.task_id);
  }

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

  getHtml(): string {
    return this.getContent("html");
  }

  getMarkdown(): string {
    return this.getContent("markdown");
  }
}

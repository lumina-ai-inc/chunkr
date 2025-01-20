import { Chunkr } from "../Chunkr";
import { Status } from "./Configuration";

export class TaskResponse {
  private chunkr: Chunkr;
  public taskId: string;
  public status: Status;
  public result?: any;
  public error?: string;

  constructor(data: any, chunkr: Chunkr) {
    this.chunkr = chunkr;
    this.taskId = data.task_id;
    this.status = data.status;
    this.result = data.result;
    this.error = data.error;
  }

  async poll(interval: number = 1000): Promise<TaskResponse> {
    while (
      this.status === Status.STARTING ||
      this.status === Status.PROCESSING
    ) {
      await new Promise((resolve) => setTimeout(resolve, interval));
      const response = await this.chunkr.getTask(this.taskId);
      this.status = response.status;
      this.result = response.result;
      this.error = response.error;
    }

    if (this.status === Status.FAILED) {
      throw new Error(
        this.error || "Task failed without specific error message",
      );
    }

    return this;
  }

  async cancel(): Promise<void> {
    await this.chunkr.cancelTask(this.taskId);
    this.status = Status.CANCELLED;
  }

  async delete(): Promise<void> {
    await this.chunkr.deleteTask(this.taskId);
  }
}

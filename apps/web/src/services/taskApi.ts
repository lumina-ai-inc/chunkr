import axiosInstance from "./axios.config";
import { TaskResponse } from "../models/task.model";

export async function getTask(taskId: string): Promise<TaskResponse> {
  const { data } = await axiosInstance.get(`/api/task/${taskId}`);
  return data;
}

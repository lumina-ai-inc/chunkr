import axiosInstance from "./axios.config";
import { TaskResponse } from "../models/task.model";

export async function getTask(taskId: string): Promise<TaskResponse> {
  const { data } = await axiosInstance.get(`/api/v1/task/${taskId}`);
  return data;
}

export async function getTasks(
  page: number,
  limit: number
): Promise<TaskResponse[]> {
  const { data } = await axiosInstance.get<TaskResponse[]>(
    "/api/v1/tasks?page=" + page + "&limit=" + limit
  );

  return data;
}

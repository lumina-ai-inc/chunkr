import axiosInstance from "./axios.config";
import { TaskResponse } from "../models/taskResponse.model";

export async function getTask(taskId: string): Promise<TaskResponse> {
  const { data } = await axiosInstance.get(`/api/v1/task/${taskId}`);
  return data;
}

export async function getTasks(
  page?: number,
  limit?: number,
  start?: string,
  end?: string
): Promise<TaskResponse[]> {
  const params = new URLSearchParams();
  if (page) params.append("page", page.toString());
  if (limit) params.append("limit", limit.toString());
  if (start) params.append("start", start);
  if (end) params.append("end", end);

  const url = `/api/v1/tasks?${params.toString()}`;
  const { data } = await axiosInstance.get<TaskResponse[]>(url);
  return data;
}

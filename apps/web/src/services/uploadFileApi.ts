import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/task.model";
import { BoundingBoxes } from "../models/chunk.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const formData = new FormData();
  for (const [key, value] of Object.entries(payload)) {
    if (value instanceof File) {
      formData.append(key, value, value.name);
    } else {
      formData.append(key, value);
    }
  }
  console.log(formData);
  const { data } = await axiosInstance.post("/api/task", formData);
  return data;
}

export async function getTask(taskId: string): Promise<TaskResponse> {
  const { data } = await axiosInstance.get(`/api/task/${taskId}`);
  return data;
}

export async function getFile(fileUrl: string): Promise<BoundingBoxes> {
  const { data } = await axiosInstance.get(fileUrl);
  return data;
}

export async function getPDF(fileUrl: string): Promise<File> {
  const { data } = await axiosInstance.get(fileUrl, { responseType: "blob" });
  return new File([data], "document.pdf", { type: data.type });
}

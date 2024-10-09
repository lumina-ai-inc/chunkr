import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/task.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const formData = new FormData();
  for (const [key, value] of Object.entries(payload)) {
    if (value instanceof File) {
      formData.append(key, value, value.name);
    } else {
      formData.append(key, value);
    }
  }
  const { data } = await axiosInstance.post("/api/v1/task", formData, {
    timeout: 5000,
  });
  return data;
}

export async function fetchFileFromSignedUrl(signedUrl: string): Promise<Blob> {
  const response = await fetch(signedUrl);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.blob();
}

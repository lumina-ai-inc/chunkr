import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/taskResponse.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const { data } = await axiosInstance.post<TaskResponse>(
    "/api/v1/task/parse",
    payload,
    {
      headers: { "Content-Type": "application/json" },
      timeout: 300000,
    }
  );
  return data;
}

export async function fetchFileFromSignedUrl(signedUrl: string): Promise<Blob> {
  const response = await fetch(signedUrl);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.blob();
}

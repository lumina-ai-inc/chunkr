import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/taskResponse.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const formData = new FormData();
  for (const [key, value] of Object.entries(payload)) {
    if (value === null || value === undefined) continue;

    if (value instanceof File) {
      formData.append(key, value, value.name);
    } else {
      // Convert object to JSON and create a Blob/File with application/json type
      const jsonBlob = new Blob([JSON.stringify(value)], {
        type: "application/json",
      });
      const jsonFile = new File([jsonBlob], `${key}.json`, {
        type: "application/json",
      });
      formData.append(key, jsonFile);
    }
  }
  const { data } = await axiosInstance.post("/api/v1/task", formData, {
    timeout: 300000,
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

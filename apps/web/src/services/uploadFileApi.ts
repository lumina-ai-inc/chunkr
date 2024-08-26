import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/task.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const hostname = import.meta.env.VITE_API_URL;
  const key = import.meta.env.VITE_API_KEY;
  const url = `${hostname}/api/task`;
  const apiKey = `${key}`;

  console.log("API Payload:", payload);
  const formData = new FormData();
  for (const [key, value] of Object.entries(payload)) {
    if (value instanceof File) {
      formData.append(key, value, value.name);
    } else {
      formData.append(key, value);
    }
  }

  const response = await fetch(url, {
    method: "POST",
    body: formData,
    headers: {
      "x-api-key": apiKey,
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  console.log("API Response:", data);
  return data;
}

export async function getTask(taskId: string): Promise<TaskResponse> {
  const hostname = import.meta.env.VITE_API_URL;
  const key = import.meta.env.VITE_API_KEY;
  const url = `${hostname}/api/task/${taskId}`;
  console.log("Task URL:", url);
  const response = await fetch(url, {
    method: "GET",
    headers: {
      "x-api-key": key,
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  console.log("API Status Response:", data);
  return data;
}

export async function getFile(fileUrl: string): Promise<string> {
  const url = `${fileUrl}`;
  console.log("File URL:", url);
  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  console.log("File Data:", data);
  return data;
}

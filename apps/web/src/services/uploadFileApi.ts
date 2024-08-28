import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/task.model";
import { BoundingBoxes } from "../models/chunk.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const hostname = import.meta.env.VITE_API_URL;
  const key = import.meta.env.VITE_API_KEY;
  const url = `${hostname}/api/task`;
  const apiKey = `${key}`;

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

  return data;
}

export async function getFile(fileUrl: string): Promise<BoundingBoxes> {
  const url = `${fileUrl}`;

  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data;
}

export async function getPDF(fileUrl: string): Promise<File> {
  const url = `${fileUrl}`;

  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.blob();

  return new File([data], "document.pdf", { type: data.type });
}

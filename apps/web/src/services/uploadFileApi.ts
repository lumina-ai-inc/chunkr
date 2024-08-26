import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/task.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  const url = `http://localhost:8000/api/task`;

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
      "x-api-key": "lu_H5kAE78qF4z_K1KCH1vcNZjN6gMeeuAJU9TD7crQChzwv",
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  console.log("API Response:", data);
  return data;
}

export async function healthCheck() {
  const url = `http://localhost:8000/health`;
  const response = await fetch(url);
  return response.json();
}

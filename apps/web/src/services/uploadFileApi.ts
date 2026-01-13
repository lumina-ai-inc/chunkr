import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";
import { TaskResponse } from "../models/taskResponse.model";

export async function uploadFile(payload: UploadForm): Promise<TaskResponse> {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'uploadFileApi.ts:6',message:'API call starting',data:{endpoint:'/api/v1/task/parse',fileName:payload.file_name,segmentationStrategy:payload.segmentation_strategy},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2-H3'})}).catch(()=>{});
  // #endregion
  const { data } = await axiosInstance.post<TaskResponse>(
    "/api/v1/task/parse",
    payload,
    {
      headers: { "Content-Type": "application/json" },
      timeout: 300000,
    }
  );
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'uploadFileApi.ts:14',message:'API call succeeded',data:{taskId:data.task_id,status:data.status},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H3'})}).catch(()=>{});
  // #endregion
  return data;
}

export async function fetchFileFromSignedUrl(signedUrl: string): Promise<Blob> {
  const response = await fetch(signedUrl);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.blob();
}

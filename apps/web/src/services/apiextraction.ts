import axios from "axios";
import { ExtractPayload } from "../models/extract.model";

export async function extractFile(payload: ExtractPayload): Promise<any> {
  const url = `${process.env.API_BASE_URL}/api/task`;

  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("model", payload.model);

  if (payload.tableOcr) {
    formData.append("table_ocr", payload.tableOcr);
  }

  if (payload.tableOcrModel) {
    formData.append("table_ocr_model", payload.tableOcrModel);
  }

  try {
    const response = await axios.post(url, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        // Add any other necessary headers, e.g., authorization
      },
    });

    return response.data;
  } catch (error) {
    console.error("Error extracting file:", error);
    throw error;
  }
}

import axios from "axios";
import { ExtractPayload } from "../models/extract.model";

export async function extractFile(payload: ExtractPayload): Promise<any> {
  const url = `${process.env.API_BASE_URL}/api/task`;

  const formData = new FormData();
  formData.append("file", payload.file);

  // Create a separate object for JSON data
  const jsonData = {
    model: payload.model,
    table_ocr: payload.tableOcr,
    table_ocr_model: payload.tableOcrModel,
  };

  // Append the JSON data as a blob
  formData.append(
    "json",
    new Blob([JSON.stringify(jsonData)], {
      type: "application/json",
    })
  );
  console.log(formData);
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

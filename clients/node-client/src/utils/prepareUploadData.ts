import { createReadStream, statSync } from "fs";
import { Configuration } from "../models/Configuration";
import FormData from "form-data";
import { ReadStream } from "fs";
import axios from "axios";

// Enhanced FileInput type
export type FileInput =
  | string // File path or URL or base64
  | Buffer // Raw buffer
  | ReadStream // File stream
  | { url: string } // URL object
  | { base64: string } // Base64 object
  | { imageData: any }; // Image data (from canvas/image manipulation)

export async function prepareUploadData(
  file: FileInput | null,
  config?: Configuration,
): Promise<FormData> {
  const formData = new FormData();

  if (file) {
    if (typeof file === "string") {
      // Check if it's a URL
      if (file.startsWith("http://") || file.startsWith("https://")) {
        formData.append("file_url", file);
      }
      // Check if it's base64
      else if (file.startsWith("data:")) {
        const base64Data = file.split(",")[1];
        const buffer = Buffer.from(base64Data, "base64");
        formData.append("file", buffer, { filename: "document" });
      }
      // Treat as file path
      else {
        formData.append("file", createReadStream(file), {
          filename: file.split("/").pop(),
          knownLength: statSync(file).size,
        });
      }
    }
    // Handle object inputs
    else if (typeof file === "object") {
      if ("url" in file) {
        formData.append("file_url", file.url);
      } else if ("base64" in file) {
        const buffer = Buffer.from(file.base64, "base64");
        formData.append("file", buffer, { filename: "document" });
      } else if ("imageData" in file) {
        // Handle image data (e.g., from Canvas)
        const buffer = Buffer.from(file.imageData);
        formData.append("file", buffer, { filename: "image" });
      } else if (Buffer.isBuffer(file)) {
        // Regular buffer
        formData.append("file", file, { filename: "document" });
      } else if (file instanceof ReadStream) {
        // ReadStream
        formData.append("file", file);
      }
    }
  }

  if (config) {
    formData.append("config", JSON.stringify(config));
  }

  return formData;
}

// Helper function to download from URL if needed
async function downloadFile(url: string): Promise<Buffer> {
  const response = await axios.get(url, { responseType: "arraybuffer" });
  return Buffer.from(response.data);
}

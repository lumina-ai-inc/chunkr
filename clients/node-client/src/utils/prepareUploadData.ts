import { createReadStream, statSync } from "fs";
import { Configuration } from "../models/Configuration";
import FormData from "form-data";
import { ReadStream } from "fs";

export type FileInput = string | Buffer | ReadStream;

async function prepareFile(
  file: FileInput,
): Promise<[string, Buffer | ReadStream]> {
  // Handle URLs
  if (
    typeof file === "string" &&
    (file.startsWith("http://") || file.startsWith("https://"))
  ) {
    try {
      // Download the file
      const response = await fetch(file);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Convert to buffer and process it like a regular buffer input
      const buffer = Buffer.from(await response.arrayBuffer());

      // Get filename from URL or default to "document.pdf"
      let filename = "document.pdf";
      try {
        const url = new URL(file);
        const pathName = decodeURIComponent(url.pathname);
        const pathSegments = pathName.split("/");
        const urlFilename = pathSegments[pathSegments.length - 1];
        if (urlFilename && urlFilename.includes(".")) {
          filename = urlFilename;
        }
      } catch (e) {
        // Keep default filename if URL parsing fails
      }

      // Process as a buffer
      return [filename, buffer];
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      throw new Error(`Failed to download remote file: ${errorMessage}`);
    }
  }

  // Handle base64 strings
  if (typeof file === "string" && file.includes(";base64,")) {
    const [header, base64Data] = file.split(",", 2);
    const buffer = Buffer.from(base64Data, "base64");

    // Extract MIME type and map to extension
    const mimeType = header.split(":")[1]?.split(";")[0].toLowerCase();
    const mimeToExt: { [key: string]: string } = {
      "application/pdf": "pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        "docx",
      "application/msword": "docx",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        "pptx",
      "application/vnd.ms-powerpoint": "pptx",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        "xlsx",
      "application/vnd.ms-excel": "xlsx",
      "image/jpeg": "jpg",
      "image/png": "png",
      "image/jpg": "jpg",
    };

    const ext = mimeToExt[mimeType];

    return [`document.${ext}`, buffer];
  }

  // Handle file paths
  if (typeof file === "string" && !file.startsWith("http")) {
    try {
      // Verify file exists and get stats
      statSync(file);
      const stream = createReadStream(file);
      const pathSegments = file.split(/[\\/]/);
      const filename = pathSegments[pathSegments.length - 1];
      return [filename, stream];
    } catch (error) {
      throw new Error(`File not found: ${file}`);
    }
  }

  // Handle Buffer
  if (Buffer.isBuffer(file)) {
    return ["document", file];
  }

  // Handle ReadStream
  if (file instanceof ReadStream) {
    const filename = file.path?.toString().split(/[\\/]/).pop() || "document";
    return [filename, file];
  }

  throw new TypeError(`Unsupported file type: ${typeof file}`);
}

export async function prepareUploadData(
  file: FileInput | null,
  config?: Configuration,
): Promise<FormData> {
  const formData = new FormData();

  if (file) {
    const [filename, fileData] = await prepareFile(file);
    formData.append("file", fileData, filename); // Changed to pass filename directly
  }

  if (config) {
    Object.entries(config).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, JSON.stringify(value), {
          contentType: "application/json",
        });
      }
    });
  }

  return formData;
}

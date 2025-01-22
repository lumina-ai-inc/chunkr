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
    const response = await fetch(file);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    let filename = "downloaded_file";

    // Try to get filename from Content-Disposition
    const contentDisposition = response.headers.get("content-disposition");
    if (contentDisposition?.includes("filename=")) {
      filename = contentDisposition
        .split("filename=")[1]
        .replace(/["']/g, "")
        .trim();
    } else {
      // Try to get filename from URL
      try {
        const url = new URL(file);
        const pathSegments = url.pathname.split("/");
        const lastSegment = pathSegments[pathSegments.length - 1];
        if (lastSegment) {
          filename = decodeURIComponent(lastSegment);
        }
      } catch (e) {
        // Keep default filename if URL parsing fails
      }
    }

    // Sanitize filename
    filename = filename
      .replace(/[<>:"/\\|?*%]/g, "_")
      .replace(/\s+/g, "_")
      .replace(/^[._]+|[._]+$/g, "")
      .slice(0, 255);

    const buffer = Buffer.from(await response.arrayBuffer());
    return [filename, buffer];
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
      "application/msword": "doc",
      "image/jpeg": "jpg",
      "image/png": "png",
      "image/jpg": "jpg",
      // Add more MIME types as needed
    };

    const ext = mimeToExt[mimeType] || "bin";
    return [`file.${ext}`, buffer];
  }

  // Handle file paths
  if (typeof file === "string" && !file.startsWith("http")) {
    const stream = createReadStream(file);
    return [file, stream]; // Use the original file path/name directly
  }

  // Handle Buffer
  if (Buffer.isBuffer(file)) {
    return ["document", file];
  }

  // Handle ReadStream
  if (file instanceof ReadStream) {
    return [file.path?.toString() || "document", file]; // Use the original path if available
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
    formData.append("file", fileData, { filename });
  }

  if (config) {
    formData.append("config", JSON.stringify(config));
  }

  return formData;
}

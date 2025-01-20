import { createReadStream, statSync } from "fs";
import { Configuration } from "../models/Configuration";
import FormData from "form-data";
import { ReadStream } from "fs";

export type FileInput = string | Buffer | ReadStream;

export async function prepareUploadData(
  file: FileInput | null,
  config?: Configuration,
): Promise<FormData> {
  const formData = new FormData();

  if (file) {
    if (typeof file === "string") {
      // File path
      formData.append("file", createReadStream(file), {
        filename: file.split("/").pop(),
        knownLength: statSync(file).size,
      });
    } else if (Buffer.isBuffer(file)) {
      // Buffer
      formData.append("file", file, {
        filename: "document",
      });
    } else {
      // ReadStream
      formData.append("file", file);
    }
  }

  if (config) {
    formData.append("config", JSON.stringify(config));
  }

  return formData;
}

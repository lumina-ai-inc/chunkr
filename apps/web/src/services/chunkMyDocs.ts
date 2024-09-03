import { getFile, getPDF } from "./uploadFileApi";
import { BoundingBoxes } from "../models/chunk.model";

export async function retrieveFileContent(
  fileUrl: string
): Promise<BoundingBoxes> {
  const fileContent = await getFile(fileUrl);
  return fileContent;
}

export async function fetchPdfFile(fileUrl: string): Promise<File> {
  const fileContent = await getPDF(fileUrl);
  return fileContent;
}

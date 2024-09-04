import { Chunks } from "../models/chunk.model";
import { fetchFileFromSignedUrl } from "./uploadFileApi";

export async function getChunks(
  fileUrl: string
): Promise<Chunks> {
  const fileBlob = await fetchFileFromSignedUrl(fileUrl);
  const fileText = await fileBlob.text();
  const fileContent: Chunks = JSON.parse(fileText);
  return fileContent;
}

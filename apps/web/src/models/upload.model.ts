export enum Model {
  Research = "Research",
  Fast = "Fast",
  HighQuality = "HighQuality",
}

export interface UploadForm {
  file: File;
  model: Model;
  ocr_strategy: "auto" | "all" | "off";
  target_chunk_length?: number;
}

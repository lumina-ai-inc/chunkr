export enum Model {
  Research = "Research",
  Fast = "Fast",
  HighQuality = "HighQuality",
}

export interface UploadForm {
  file: File;
  model: Model;
  ocr_strategy: "Auto" | "All" | "Off";
  target_chunk_length?: number;
}

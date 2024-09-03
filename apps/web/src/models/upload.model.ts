export enum Model {
  Research = "Research",
  Fast = "Fast",
  HighQuality = "HighQuality",
}

export enum TableOcr {
  HTML = "HTML",
  JSON = "JSON",
}

export enum TableOcrModel {
  EasyOcr = "EasyOcr",
  Tesseract = "Tesseract",
}

export interface UploadForm {
  file: File;
  model: Model;
  target_chunk_length?: number;
}

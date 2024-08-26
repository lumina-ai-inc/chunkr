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

export interface ExtractPayload {
  file: File;
  model: Model;
  tableOcr?: TableOcr;
  tableOcrModel?: TableOcrModel;
}

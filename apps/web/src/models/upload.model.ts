import {
  ChunkProcessing,
  JsonSchema,
  OcrStrategy,
  SegmentProcessing,
  SegmentationStrategy,
  Pipeline,
} from "./taskConfig.model";

export interface UploadForm {
  /** The file to be uploaded */
  file: File;

  /** Optional chunk processing configuration */
  chunk_processing?: ChunkProcessing;

  /** Time until task deletion in seconds */
  expires_in?: number;

  /** Use high-res images for processing */
  high_resolution?: boolean;

  /** Schema for structured data extraction */
  json_schema?: JsonSchema;

  /** OCR processing strategy */
  ocr_strategy?: OcrStrategy;

  /** Segment processing configuration */
  segment_processing?: SegmentProcessing;

  /** Strategy for document segmentation */
  segmentation_strategy?: SegmentationStrategy;

  /** Pipeline to run after processing */
  pipeline?: Pipeline;
}

import {
  ChunkProcessing,
  OcrStrategy,
  SegmentProcessing,
  SegmentationStrategy,
  Pipeline,
} from "./taskConfig.model";
import { WhenEnabled } from "../config/env.config";

export interface UploadForm {
  /** Base64 data payload (no “data:” prefix) or a public URL */
  file: string;
  /** original filename for server‐side reference */
  file_name: string;

  /** Optional chunk processing configuration */
  chunk_processing?: ChunkProcessing;

  /** Time until task deletion in seconds */
  expires_in?: number;

  /** Use high-res images for processing */
  high_resolution?: boolean;

  /** OCR processing strategy */
  ocr_strategy?: OcrStrategy;

  /** Segment processing configuration */
  segment_processing?: SegmentProcessing;

  /** Strategy for document segmentation */
  segmentation_strategy?: SegmentationStrategy;

  /** Pipeline to run after processing */
  pipeline?: WhenEnabled<"pipeline", Pipeline>;
}

import {
  OcrStrategy,
  SegmentationStrategy,
  ChunkProcessing,
  SegmentProcessing,
} from "./taskConfig.model";

export enum Status {
  Starting = "Starting",
  Processing = "Processing",
  Succeeded = "Succeeded",
  Failed = "Failed",
  Cancelled = "Cancelled",
}

export interface Chunk {
  chunk_id: string;
  chunk_length: number;
  segments: Segment[];
}

export interface Output {
  pdf_url: string | null;
  file_name: string | null;
  page_count: number | null;
  chunks: Chunk[];
}

export interface TaskResponse {
  task_id: string;
  status: Status;
  created_at: string;
  finished_at: string | null;
  expires_at: string | null;
  message: string;
  output: Output | null;
  task_url: string | null;
  configuration: Configuration;
}

export interface Configuration {
  input_file_url: string | null;
  ocr_strategy: OcrStrategy;
  segmentation_strategy: SegmentationStrategy;
  high_resolution?: boolean;
  chunk_processing?: ChunkProcessing;
  segment_processing?: SegmentProcessing;
}

// Define a JSON value type that matches serde_json::Value capabilities

export interface BoundingBox {
  height: number;
  left: number;
  top: number;
  width: number;
}

export interface OCRResult {
  bbox: BoundingBox;
  confidence: number | null;
  text: string;
}

export interface Segment {
  bbox: BoundingBox;
  confidence: number | null;
  content: string;
  html: string | null;
  image: string | null;
  markdown: string | null;
  llm: string | null;
  ocr: OCRResult[];
  page_height: number;
  page_number: number;
  page_width: number;
  segment_id: string;
  segment_type: SegmentType;
}

export enum SegmentType {
  Caption = "Caption",
  Footnote = "Footnote",
  Formula = "Formula",
  ListItem = "ListItem",
  Page = "Page",
  PageFooter = "PageFooter",
  PageHeader = "PageHeader",
  Picture = "Picture",
  SectionHeader = "SectionHeader",
  Table = "Table",
  Text = "Text",
  Title = "Title",
}

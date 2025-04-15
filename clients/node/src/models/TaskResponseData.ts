import { Configuration } from "./Configuration";

export interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface OCRResult {
  bbox: BoundingBox;
  text: string;
  confidence?: number | null;
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

export interface Segment {
  segment_id: string;
  bbox: BoundingBox;
  page_number: number;
  page_width: number;
  page_height: number;
  content: string;
  segment_type: SegmentType;
  ocr?: OCRResult[];
  image?: string;
  html?: string;
  markdown?: string;
}

export interface Chunk {
  chunk_id: string;
  chunk_length: number;
  segments: Segment[];
}

export interface Output {
  file_name: string | null;
  pdf_url: string | null;
  page_count: number | null;
  chunks: Chunk[];
}

export enum Status {
  STARTING = "Starting",
  PROCESSING = "Processing",
  SUCCEEDED = "Succeeded",
  FAILED = "Failed",
  CANCELLED = "Cancelled",
}

export interface TaskResponseData {
  task_id: string;
  status: Status;
  configuration: Configuration;
  created_at: string;
  finished_at: string | null;
  expires_at: string | null;
  message: string;
  output: Output | null;
  task_url: string | null;
  error?: string;
}


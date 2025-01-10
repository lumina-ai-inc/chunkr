import {
  OcrStrategy,
  SegmentationStrategy,
  ChunkProcessing,
  SegmentProcessing,
  JsonSchema,
} from "./newTask.model";

export enum Status {
  Starting = "Starting",
  Processing = "Processing",
  Succeeded = "Succeeded",
  Failed = "Failed",
}

export interface Chunk {
  chunk_id: string;
  chunk_length: number;
  segments: Segment[];
}

export interface Output {
  chunks: Chunk[];
  extracted_json: ExtractedJson | null;
}

export interface TaskResponse {
  task_id: string;
  status: Status;
  created_at: string;
  finished_at: string | null;
  expires_at: string | null;
  message: string;
  input_file_url: string | null;
  pdf_url: string | null;
  output: Output | null;
  task_url: string | null;
  configuration: Configuration;
  file_name: string | null;
  page_count: number | null;
}

export interface Configuration {
  ocr_strategy: OcrStrategy;
  segmentation_strategy: SegmentationStrategy;
  high_resolution?: boolean;
  chunk_processing?: ChunkProcessing;
  segment_processing?: SegmentProcessing;
  json_schema?: JsonSchema;
}

export interface ExtractedJson {
  title: string;
  schema_type: string;
  extracted_fields: ExtractedField[];
}

// Define a JSON value type that matches serde_json::Value capabilities
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface ExtractedField {
  name: string;
  field_type: string;
  value: JsonValue;
}

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

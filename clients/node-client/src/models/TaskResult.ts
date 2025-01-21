import { Status } from "./Configuration";

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

export interface ExtractedField {
  name: string;
  field_type: string;
  value: JsonValue;
}

export interface ExtractedJson {
  title: string;
  schema_type: string;
  extracted_fields: ExtractedField[];
}

export interface Output {
  chunks: Chunk[];
  extracted_json: ExtractedJson | null;
}

export interface TaskResult {
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
  file_name: string | null;
  page_count: number | null;
  error?: string;
}

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

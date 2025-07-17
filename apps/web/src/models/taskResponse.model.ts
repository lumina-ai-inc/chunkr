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

export enum Alignment {
  Left = "left",
  Center = "center",
  Right = "right",
  Justify = "justify",
}

export enum VerticalAlignment {
  Top = "top",
  Middle = "middle",
  Bottom = "bottom",
  Baseline = "baseline",
}

export interface Chunk {
  chunk_id: string;
  chunk_length: number;
  segment_length?: number;
  segments: Segment[];
}

export interface Output {
  pdf_url: string | null;
  file_name: string | null;
  page_count: number | null;
  pages?: Page[];
  chunks: Chunk[];
}

export interface TaskResponse {
  task_id: string;
  status: Status;
  created_at: string;
  finished_at: string | null;
  expires_at: string | null;
  started_at: string | null;
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

export interface CellStyle {
  bg_color?: string;
  text_color?: string;
  font_face?: string;
  is_bold?: boolean;
  align?: Alignment;
  valign?: VerticalAlignment;
}

export interface Cell {
  cell_id: string;
  text: string;
  range: string;
  formula?: string;
  value?: string;
  hyperlink?: string;
  style?: CellStyle;
}

export interface Page {
  image: string;
  page_number: number;
  page_height: number;
  page_width: number;
  ss_sheet_name?: string | null;
}

export interface Segment {
  bbox: BoundingBox;
  confidence: number | null;
  content: string;
  html: string;
  image: string | null;
  markdown: string;
  llm: string | null;
  ocr?: OCRResult[];
  page_height: number;
  page_number: number;
  page_width: number;
  segment_id: string;
  segment_type: SegmentType;
  segment_length?: number;
  ss_cells?: Cell[];
  ss_header_bbox?: BoundingBox;
  ss_header_ocr?: OCRResult[];
  ss_header_text?: string;
  ss_header_range?: string;
  ss_range?: string;
  ss_sheet_name?: string;
  text?: string;
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

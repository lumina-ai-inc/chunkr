export type SegmentType =
  | "Title"
  | "Section header"
  | "Text"
  | "List item"
  | "Table"
  | "Picture"
  | "Caption"
  | "Formula"
  | "Footnote"
  | "Page header"
  | "Page footer";

export interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface OCRResult {
  bbox: BoundingBox;
  text: string;
  confidence: number;
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
  segments: Segment[];
  chunk_length: number;
}

export type Chunks = Chunk[];

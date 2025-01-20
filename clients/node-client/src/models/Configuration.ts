export enum GenerationStrategy {
  LLM = "LLM",
  AUTO = "Auto",
}

export enum CroppingStrategy {
  ALL = "All",
  AUTO = "Auto",
}

export enum OcrStrategy {
  ALL = "All",
  AUTO = "Auto",
}

export enum SegmentationStrategy {
  LAYOUT_ANALYSIS = "LayoutAnalysis",
  PAGE = "Page",
}

export enum Status {
  STARTING = "Starting",
  PROCESSING = "Processing",
  SUCCEEDED = "Succeeded",
  FAILED = "Failed",
  CANCELLED = "Cancelled",
}

export interface GenerationConfig {
  html?: GenerationStrategy;
  llm?: string;
  markdown?: GenerationStrategy;
  crop_image?: CroppingStrategy;
}

export interface SegmentProcessing {
  title?: GenerationConfig;
  section_header?: GenerationConfig;
  text?: GenerationConfig;
  list_item?: GenerationConfig;
  table?: GenerationConfig;
  picture?: GenerationConfig;
  caption?: GenerationConfig;
  formula?: GenerationConfig;
  footnote?: GenerationConfig;
  page_header?: GenerationConfig;
  page_footer?: GenerationConfig;
  page?: GenerationConfig;
}

export interface ChunkProcessing {
  target_length?: number;
}

export interface Property {
  name: string;
  prop_type: string;
  description?: string;
  default?: string;
}

export interface JsonSchema {
  title: string;
  properties: Property[];
}

export interface Configuration {
  chunk_processing?: ChunkProcessing;
  expires_in?: number;
  high_resolution?: boolean;
  json_schema?: JsonSchema;
  ocr_strategy?: OcrStrategy;
  segment_processing?: SegmentProcessing;
  segmentation_strategy?: SegmentationStrategy;
}

/**
 * OCR Strategy options for document processing
 * - `All`: Processes all pages with OCR. (Latency penalty: ~0.5 seconds per page)
 * - `Auto`: Selectively applies OCR only to pages with missing or low-quality text
 */
export enum OcrStrategy {
  All = "All",
  Auto = "Auto",
}

/**
 * Controls the segmentation strategy
 * - `LayoutAnalysis`: Analyzes pages for layout elements using bounding boxes
 * - `Page`: Treats each page as a single segment
 */
export enum SegmentationStrategy {
  LayoutAnalysis = "LayoutAnalysis",
  Page = "Page",
}

/**
 * Controls the setting for the chunking and post-processing of each chunk.
 */
export interface ChunkProcessing {
  /**
   * Whether to ignore headers and footers when chunking.
   * @default
   */
  ignore_headers_and_footers?: boolean;

  /**
   * The target number of words in each chunk.
   * If 0, each chunk will contain a single segment.
   * @default 512
   */
  target_length: number;
}

/** Controls how content should be generated */
export enum GenerationStrategy {
  LLM = "LLM",
  Auto = "Auto",
}

/** Controls when images should be cropped */
export enum CroppingStrategy {
  All = "All",
  Auto = "Auto",
}

/** Configuration for LLM-based generation */
export interface LlmConfig {
  model: string;
  prompt: string;
  temperature: number;
}

/** Base configuration for automatic content generation */
export interface SegmentProcessingConfig {
  crop_image: CroppingStrategy;
  html: GenerationStrategy;
  llm?: LlmConfig;
  markdown: GenerationStrategy;
}

/**
 * Controls the post-processing of each segment type.
 * Allows you to generate HTML and Markdown from chunkr models for each segment type.
 * By default, the HTML and Markdown are generated manually using the segmentation information except for `Table` and `Formula`.
 * You can optionally configure custom LLM prompts and models to generate an additional `llm` field
 * with LLM-processed content for each segment type.
 */
export interface SegmentProcessing {
  Title: SegmentProcessingConfig;
  SectionHeader: SegmentProcessingConfig;
  Text: SegmentProcessingConfig;
  ListItem: SegmentProcessingConfig;
  Table: SegmentProcessingConfig;
  Picture: SegmentProcessingConfig;
  Caption: SegmentProcessingConfig;
  Formula: SegmentProcessingConfig;
  Footnote: SegmentProcessingConfig;
  PageHeader: SegmentProcessingConfig;
  PageFooter: SegmentProcessingConfig;
  Page: SegmentProcessingConfig;
}

/**
 * Represents a property in the JSON schema
 */
export interface Property {
  /** The identifier for the property in the extracted data */
  name: string;

  /**
   * A human-readable title for the property.
   * This is optional and can be used to increase the accuracy of the extraction.
   */
  title?: string;

  /**
   * The data type of the property
   */
  type: string;

  /**
   * A description of what the property represents.
   * This is optional and can be used increase the accuracy of the extraction.
   * Available for string, int, float, bool, list, object.
   */
  description?: string;

  /**
   * The default value for the property if no data is extracted
   */
  default?: string;
}

/**
 * The JSON schema to be used for structured extraction
 */
export interface JsonSchema {
  /** The title of the JSON schema. This can be used to identify the schema */
  title: string;

  /** The properties of the JSON schema. Each property is a field to be extracted from the document */
  properties: Property[];

  /**
   * @deprecated
   * The type of the JSON schema
   */
  schema_type?: string;
}

import { WhenEnabled } from "../config/env.config";

export interface UploadFormData {
  /** Optional chunk processing configuration */
  chunk_processing?: ChunkProcessing;

  /**
   * The number of seconds until task is deleted.
   * Expired tasks can **not** be updated, polled or accessed via web interface.
   */
  expires_in?: number;

  /** The file to be uploaded */
  file: File;

  /**
   * Whether to use high-resolution images for cropping and post-processing.
   * (Latency penalty: ~7 seconds per page)
   * @default false
   */
  high_resolution?: boolean;

  /** Optional JSON schema configuration for structured data extraction */
  json_schema?: JsonSchema;

  /**
   * OCR strategy to use for document processing
   * @default "Auto"
   */
  ocr_strategy?: OcrStrategy;

  /** Optional segment processing configuration */
  segment_processing?: SegmentProcessing;

  /**
   * Segmentation strategy to use
   * @default "LayoutAnalysis"
   */
  segmentation_strategy?: SegmentationStrategy;

  /**
   * @deprecated Use `chunk_processing` instead
   * The target chunk length to be used for chunking.
   * If 0, each chunk will contain a single segment.
   * @default 512
   */
  target_chunk_length?: number;

  /** Pipeline to run after processing */
  pipeline?: WhenEnabled<"pipeline", Pipeline>;
}

export enum Pipeline {
  Azure = "Azure",
}

const DEFAULT_SEGMENT_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.Auto,
  markdown: GenerationStrategy.Auto,
};

const DEFAULT_TABLE_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
};

const DEFAULT_FORMULA_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
};

const DEFAULT_PICTURE_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.All,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
};

export const DEFAULT_SEGMENT_PROCESSING: SegmentProcessing = {
  Text: { ...DEFAULT_SEGMENT_CONFIG },
  Title: { ...DEFAULT_SEGMENT_CONFIG },
  SectionHeader: { ...DEFAULT_SEGMENT_CONFIG },
  ListItem: { ...DEFAULT_SEGMENT_CONFIG },
  Table: { ...DEFAULT_TABLE_CONFIG },
  Picture: { ...DEFAULT_PICTURE_CONFIG },
  Caption: { ...DEFAULT_SEGMENT_CONFIG },
  Formula: { ...DEFAULT_FORMULA_CONFIG },
  Footnote: { ...DEFAULT_SEGMENT_CONFIG },
  PageHeader: { ...DEFAULT_SEGMENT_CONFIG },
  PageFooter: { ...DEFAULT_SEGMENT_CONFIG },
  Page: { ...DEFAULT_SEGMENT_CONFIG },
};

export const DEFAULT_UPLOAD_CONFIG: UploadFormData = {
  chunk_processing: { target_length: 512, ignore_headers_and_footers: true },
  high_resolution: false,
  ocr_strategy: OcrStrategy.All,
  segmentation_strategy: SegmentationStrategy.LayoutAnalysis,
  segment_processing: DEFAULT_SEGMENT_PROCESSING,
  json_schema: undefined, // or some default schema if needed
  file: new File([], ""),
  pipeline: undefined as WhenEnabled<"pipeline", Pipeline>, // Default pipeline
};

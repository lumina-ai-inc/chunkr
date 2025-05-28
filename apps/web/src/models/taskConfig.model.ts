import { WhenEnabled } from "../config/env.config";

export enum Tokenizer {
  Word = "Word",
  CL100K_BASE = "Cl100kBase",
  XLM_ROBERTA_BASE = "XlmRobertaBase",
  BERT_BASE_UNCASED = "BertBaseUncased",
}

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

  /**
   * Which tokenizer to use when splitting into chunks.
   * - an object with an `Enum` key containing one of the predefined enums below, or any Hugging Face ID string.
   */
  tokenizer?: { Enum: Tokenizer | string };
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

export enum EmbedSource {
  MARKDOWN = "Markdown",
  HTML = "HTML",
  LLM = "LLM",
}

/** Base configuration for automatic content generation */
export interface SegmentProcessingConfig {
  crop_image: CroppingStrategy;
  html: GenerationStrategy;
  llm?: string;
  markdown: GenerationStrategy;
  embed_sources?: EmbedSource[];
  extended_context: boolean;
}

/**
 * Controls the post-processing of each segment type.
 * Allows you to generate HTML and Markdown from chunkr models for each segment type.
 * By default, the HTML and Markdown are generated manually using the segmentation information except for `Table` and `Formula`.
 * You can optionally configure custom LLM prompts and models to generate an additional `llm` field
 * with LLM-processed content for each segment type.
 */
export interface SegmentProcessing {
  Caption: SegmentProcessingConfig;
  Formula: SegmentProcessingConfig;
  Footnote: SegmentProcessingConfig;
  ListItem: SegmentProcessingConfig;
  Page: SegmentProcessingConfig;
  PageFooter: SegmentProcessingConfig;
  PageHeader: SegmentProcessingConfig;
  Picture: SegmentProcessingConfig;
  SectionHeader: SegmentProcessingConfig;
  Table: SegmentProcessingConfig;
  Text: SegmentProcessingConfig;
  Title: SegmentProcessingConfig;
}

export enum FallbackStrategyType {
  Default = "Default",
  Model = "Model",
}

/**
 * Defines the fallback strategy for LLM processing.
 * - `{ None: null }`: No fallback strategy.
 * - `{ Default: null }`: Use the system default fallback.
 * - `{ Model: string }`: Use the specified model ID as fallback.
 */
export type FallbackStrategy =
  | { [FallbackStrategyType.Default]: null }
  | { [FallbackStrategyType.Model]: string }; // string is the model_id

export interface LlmProcessing {
  /** ID of the model to use. undefined ⇒ system default */
  model_id?: string;
  /** what to do if primary fails */
  fallback_strategy?: FallbackStrategy;
  /** max tokens to generate */
  max_completion_tokens?: number;
  /** randomness (0.0 = deterministic) */
  temperature?: number;
}

/**
 * Controls how errors are handled during processing.
 * - `Fail`: Stops processing and fails the task when any error occurs.
 * - `Continue`: Attempts to continue processing despite non-critical errors (e.g., LLM refusals).
 */
export enum ErrorHandling {
  Fail = "Fail",
  Continue = "Continue",
}

export interface UploadFormData {
  chunk_processing: ChunkProcessing;
  high_resolution: boolean;
  ocr_strategy: OcrStrategy;
  segmentation_strategy: SegmentationStrategy;
  segment_processing: SegmentProcessing;
  llm_processing?: LlmProcessing;
  pipeline?: WhenEnabled<"pipeline", Pipeline>;
  error_handling?: ErrorHandling;
}

export enum Pipeline {
  Azure = "Azure",
  Chunkr = "Chunkr",
}

const DEFAULT_SEGMENT_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.Auto,
  markdown: GenerationStrategy.Auto,
  embed_sources: [EmbedSource.MARKDOWN],
  extended_context: false,
};

const DEFAULT_TABLE_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
  embed_sources: [EmbedSource.MARKDOWN],
  extended_context: false,
};

const DEFAULT_FORMULA_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.Auto,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
  embed_sources: [EmbedSource.MARKDOWN],
  extended_context: false,
};

const DEFAULT_PICTURE_CONFIG: SegmentProcessingConfig = {
  crop_image: CroppingStrategy.All,
  html: GenerationStrategy.LLM,
  markdown: GenerationStrategy.LLM,
  embed_sources: [EmbedSource.MARKDOWN],
  extended_context: false,
};

export const DEFAULT_SEGMENT_PROCESSING: SegmentProcessing = {
  Caption: { ...DEFAULT_SEGMENT_CONFIG },
  Formula: { ...DEFAULT_FORMULA_CONFIG },
  Footnote: { ...DEFAULT_SEGMENT_CONFIG },
  ListItem: { ...DEFAULT_SEGMENT_CONFIG },
  Page: { ...DEFAULT_SEGMENT_CONFIG },
  PageFooter: { ...DEFAULT_SEGMENT_CONFIG },
  PageHeader: { ...DEFAULT_SEGMENT_CONFIG },
  Picture: { ...DEFAULT_PICTURE_CONFIG },
  SectionHeader: { ...DEFAULT_SEGMENT_CONFIG },
  Table: { ...DEFAULT_TABLE_CONFIG },
  Text: { ...DEFAULT_SEGMENT_CONFIG },
  Title: { ...DEFAULT_SEGMENT_CONFIG },
};

const DEFAULT_LLM_PROCESSING: LlmProcessing = {
  model_id: undefined,
  fallback_strategy: { [FallbackStrategyType.Default]: null },
  max_completion_tokens: undefined,
  temperature: 0.0,
};

export const DEFAULT_CHUNK_PROCESSING: ChunkProcessing = {
  target_length: 512,
  ignore_headers_and_footers: true,
  tokenizer: { Enum: Tokenizer.Word },
};

export const DEFAULT_UPLOAD_CONFIG: UploadFormData = {
  chunk_processing: DEFAULT_CHUNK_PROCESSING,
  high_resolution: true,
  ocr_strategy: OcrStrategy.All,
  segmentation_strategy: SegmentationStrategy.LayoutAnalysis,
  segment_processing: DEFAULT_SEGMENT_PROCESSING,
  llm_processing: DEFAULT_LLM_PROCESSING,
  pipeline: Pipeline.Azure as unknown as WhenEnabled<"pipeline", Pipeline>,
  error_handling: ErrorHandling.Fail,
};

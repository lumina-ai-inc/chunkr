use crate::models::chunkr::structured_extraction::JsonSchema;

use actix_multipart::form::json::Json as MPJson;
use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::{IntoParams, ToSchema};

#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct UploadForm {
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The number of seconds until task is deleted.
    /// Not recommended - as expried tasks can not be updated, polled or accessed via web interface.
    pub expires_in: Option<Text<i32>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    /// The file to be uploaded.
    pub file: TempFile,
    #[param(style = Form, value_type = Option<JsonSchema>)]
    #[schema(value_type = Option<JsonSchema>)]
    /// The JSON schema to be used for structured extraction.
    pub json_schema: Option<MPJson<JsonSchema>>,
    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>, default = "Auto")]
    pub ocr_strategy: Option<Text<OcrStrategy>>,
    #[param(style = Form, value_type = Option<SegmentProcessing>)]
    #[schema(value_type = Option<SegmentProcessing>)]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>)]
    #[schema(value_type = Option<SegmentationStrategy>, default = "LayoutAnalysis")]
    pub segmentation_strategy: Option<Text<SegmentationStrategy>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The target chunk length to be used for chunking.
    pub target_chunk_length: Option<Text<i32>>,
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString,
)]
/// Controls the Optical Character Recognition (OCR) strategy.
pub enum OcrStrategy {
    /// Processes all pages with OCR.
    /// + Highest accuracy
    /// + Consistent results
    /// - Slower processing
    All,

    /// Selectively applies OCR only to pages with missing or low-quality text.
    /// + Faster processing
    /// + Works for most documents
    /// - Does not guarantee OCR on all pages
    #[serde(alias = "Off")]
    Auto,
}

impl Default for OcrStrategy {
    fn default() -> Self {
        OcrStrategy::Auto
    }
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Display,
    EnumString,
    Eq,
    PartialEq,
    ToSql,
    FromSql,
    ToSchema,
)]
/// Controls the segmentation strategy.
pub enum SegmentationStrategy {
    /// Analyzes pages for layout elements (Table, Picture, Formula) using bounding boxes.
    /// + Fine-grained segmentation
    /// + Better chunking and segment post-processing
    /// - Latency penalty
    LayoutAnalysis,

    /// Treats each page as a single segment of type `Page`.
    /// + Faster processing
    /// - No layout element detection
    /// - Only simple chunking and segment post-processing
    Page,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
/// Controls the post-processing of each segment type.
/// Allows you to generate HTML and Markdown from chunkr models for each segment type.
/// By default, the HTML and Markdown are generated manually using the segmentation information except for `Table` and `Formula`.
/// You can optionally configure custom LLM prompts and models to generate an additional `llm` field
/// with LLM-processed content for each segment type.
pub struct SegmentProcessing {
    #[serde(rename = "Title", default)]
    pub title: SegmentProcessingConfig,
    #[serde(rename = "SectionHeader", default)]
    pub section_header: SegmentProcessingConfig,
    #[serde(rename = "Text", default)]
    pub text: SegmentProcessingConfig,
    #[serde(rename = "ListItem", default)]
    pub list_item: SegmentProcessingConfig,
    #[serde(rename = "Table", default)]
    pub table: SegmentProcessingConfigLLM,
    #[serde(rename = "Picture", default)]
    pub picture: SegmentProcessingConfig,
    #[serde(rename = "Caption", default)]
    pub caption: SegmentProcessingConfig,
    #[serde(rename = "Formula", default)]
    pub formula: SegmentProcessingConfigLLM,
    #[serde(rename = "Footnote", default)]
    pub footnote: SegmentProcessingConfig,
    #[serde(rename = "PageHeader", default)]
    pub page_header: SegmentProcessingConfig,
    #[serde(rename = "PageFooter", default)]
    pub page_footer: SegmentProcessingConfig,
    #[serde(rename = "Page", default)]
    pub page: SegmentProcessingConfig,
}

impl Default for SegmentProcessing {
    fn default() -> Self {
        Self {
            title: SegmentProcessingConfig::default(),
            section_header: SegmentProcessingConfig::default(),
            text: SegmentProcessingConfig::default(),
            list_item: SegmentProcessingConfig::default(),
            table: SegmentProcessingConfigLLM::default(),
            picture: SegmentProcessingConfig::default(),
            caption: SegmentProcessingConfig::default(),
            formula: SegmentProcessingConfigLLM::default(),
            footnote: SegmentProcessingConfig::default(),
            page_header: SegmentProcessingConfig::default(),
            page_footer: SegmentProcessingConfig::default(),
            page: SegmentProcessingConfig::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct SegmentProcessingConfig {
    #[serde(default = "default_generation_strategy")]
    #[schema(default = "Auto")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_generation_strategy")]
    #[schema(default = "Auto")]
    pub markdown: GenerationStrategy,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct SegmentProcessingConfigLLM {
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub markdown: GenerationStrategy,
}

fn default_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::Auto
}

fn default_llm_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::LLM
}

impl Default for SegmentProcessingConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::Auto,
            llm: None,
            markdown: GenerationStrategy::Auto,
        }
    }
}

impl Default for SegmentProcessingConfigLLM {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::LLM,
            llm: None,
            markdown: GenerationStrategy::LLM,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, Display, EnumString)]
pub enum GenerationStrategy {
    LLM,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct LlmConfig {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    0.0
}

use crate::models::chunkr::cropping::CroppingStrategy;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// Controls the post-processing of each segment type.
/// Allows you to generate HTML and Markdown from chunkr models for each segment type.
/// By default, the HTML and Markdown are generated manually using the segmentation information except for `Table` and `Formula`.
/// You can optionally configure custom LLM prompts and models to generate an additional `llm` field
/// with LLM-processed content for each segment type.
pub struct SegmentProcessing {
    #[serde(rename = "Title", alias = "title", default)]
    pub title: AutoGenerationConfig,
    #[serde(rename = "SectionHeader", alias = "section_header", default)]
    pub section_header: AutoGenerationConfig,
    #[serde(rename = "Text", alias = "text", default)]
    pub text: AutoGenerationConfig,
    #[serde(rename = "ListItem", alias = "list_item", default)]
    pub list_item: AutoGenerationConfig,
    #[serde(rename = "Table", alias = "table", default)]
    pub table: LlmGenerationConfig,
    #[serde(rename = "Picture", alias = "picture", default)]
    pub picture: AutoGenerationConfig,
    #[serde(rename = "Caption", alias = "caption", default)]
    pub caption: AutoGenerationConfig,
    #[serde(rename = "Formula", alias = "formula", default)]
    pub formula: LlmGenerationConfig,
    #[serde(rename = "Footnote", alias = "footnote", default)]
    pub footnote: AutoGenerationConfig,
    #[serde(rename = "PageHeader", alias = "page_header", default)]
    pub page_header: AutoGenerationConfig,
    #[serde(rename = "PageFooter", alias = "page_footer", default)]
    pub page_footer: AutoGenerationConfig,
    #[serde(rename = "Page", alias = "page", default)]
    pub page: AutoGenerationConfig,
}

impl Default for SegmentProcessing {
    fn default() -> Self {
        Self {
            title: AutoGenerationConfig::default(),
            section_header: AutoGenerationConfig::default(),
            text: AutoGenerationConfig::default(),
            list_item: AutoGenerationConfig::default(),
            table: LlmGenerationConfig::default(),
            picture: AutoGenerationConfig::default(),
            caption: AutoGenerationConfig::default(),
            formula: LlmGenerationConfig::default(),
            footnote: AutoGenerationConfig::default(),
            page_header: AutoGenerationConfig::default(),
            page_footer: AutoGenerationConfig::default(),
            page: AutoGenerationConfig::default(),
        }
    }
}

// TODO: Change to macro
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the processing and generation for the segment.
/// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
///   or `Auto` to only crop when needed for post-processing.
/// - `html` is the HTML output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
pub struct AutoGenerationConfig {
    #[serde(default = "default_cropping_strategy")]
    #[schema(value_type = CroppingStrategy, default = "Auto")]
    pub crop_image: CroppingStrategy,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub markdown: GenerationStrategy,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the processing and generation for the segment.
/// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
///   or `Auto` to only crop when needed for post-processing.
/// - `html` is the HTML output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
pub struct LlmGenerationConfig {
    #[serde(default = "default_cropping_strategy")]
    #[schema(value_type = CroppingStrategy, default = "Auto")]
    pub crop_image: CroppingStrategy,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub markdown: GenerationStrategy,
}

fn default_cropping_strategy() -> CroppingStrategy {
    CroppingStrategy::Auto
}

fn default_auto_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::Auto
}

fn default_llm_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::LLM
}

impl Default for AutoGenerationConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::Auto,
            llm: None,
            markdown: GenerationStrategy::Auto,
            crop_image: default_cropping_strategy(),
        }
    }
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::LLM,
            llm: None,
            markdown: GenerationStrategy::LLM,
            crop_image: default_cropping_strategy(),
        }
    }
}

#[derive(
    Debug,
    Serialize,
    Deserialize,
    Clone,
    ToSchema,
    Display,
    EnumString,
    ToSql,
    FromSql,
    PartialEq,
    Eq,
)]
pub enum GenerationStrategy {
    LLM,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the LLM-generated output for the segment.
pub struct LlmConfig {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    0.0
}

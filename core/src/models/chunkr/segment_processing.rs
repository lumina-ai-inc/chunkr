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
    #[serde(rename = "Title", default)]
    pub title: AutoGenerationConfig,
    #[serde(rename = "SectionHeader", default)]
    pub section_header: AutoGenerationConfig,
    #[serde(rename = "Text", default)]
    pub text: AutoGenerationConfig,
    #[serde(rename = "ListItem", default)]
    pub list_item: AutoGenerationConfig,
    #[serde(rename = "Table", default)]
    pub table: LlmGenerationConfig,
    #[serde(rename = "Picture", default)]
    pub picture: AutoGenerationConfig,
    #[serde(rename = "Caption", default)]
    pub caption: AutoGenerationConfig,
    #[serde(rename = "Formula", default)]
    pub formula: LlmGenerationConfig,
    #[serde(rename = "Footnote", default)]
    pub footnote: AutoGenerationConfig,
    #[serde(rename = "PageHeader", default)]
    pub page_header: AutoGenerationConfig,
    #[serde(rename = "PageFooter", default)]
    pub page_footer: AutoGenerationConfig,
    #[serde(rename = "Page", default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the generation for the `html`, `llm` and `markdown` fields for the segment.
/// - `html` is the HTML output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
pub struct AutoGenerationConfig {
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub markdown: GenerationStrategy,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the generation for the `html`, `llm` and `markdown` fields for the segment.
/// - `html` is the HTML output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
pub struct LlmGenerationConfig {
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub html: GenerationStrategy,
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub markdown: GenerationStrategy,
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
        }
    }
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::LLM,
            llm: None,
            markdown: GenerationStrategy::LLM,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, Display, EnumString, ToSql, FromSql)]
pub enum GenerationStrategy {
    LLM,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
pub struct LlmConfig {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    0.0
}

use crate::models::cropping::{CroppingStrategy, PictureCroppingStrategy};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// Controls the post-processing of each segment type.
/// Allows you to generate HTML and Markdown from chunkr models for each segment type.
/// By default, the HTML and Markdown are generated manually using the segmentation information except for `Table`, `Formula` and `Picture`.
/// You can optionally configure custom LLM prompts and models to generate an additional `llm` field with LLM-processed content for each segment type.
///
/// The configuration of which content sources (HTML, Markdown, LLM, Content) of the segment
/// should be included in the chunk's `embed` field and counted towards the chunk length can be configured through the `embed_sources` setting.
pub struct SegmentProcessing {
    #[serde(rename = "Title", alias = "title")]
    pub title: Option<AutoGenerationConfig>,
    #[serde(rename = "SectionHeader", alias = "section_header")]
    pub section_header: Option<AutoGenerationConfig>,
    #[serde(rename = "Text", alias = "text")]
    pub text: Option<AutoGenerationConfig>,
    #[serde(rename = "ListItem", alias = "list_item")]
    pub list_item: Option<AutoGenerationConfig>,
    #[serde(rename = "Table", alias = "table")]
    pub table: Option<LlmGenerationConfig>,
    #[serde(rename = "Picture", alias = "picture")]
    pub picture: Option<PictureGenerationConfig>,
    #[serde(rename = "Caption", alias = "caption")]
    pub caption: Option<AutoGenerationConfig>,
    #[serde(rename = "Formula", alias = "formula")]
    pub formula: Option<LlmGenerationConfig>,
    #[serde(rename = "Footnote", alias = "footnote")]
    pub footnote: Option<AutoGenerationConfig>,
    #[serde(rename = "PageHeader", alias = "page_header")]
    pub page_header: Option<AutoGenerationConfig>,
    #[serde(rename = "PageFooter", alias = "page_footer")]
    pub page_footer: Option<AutoGenerationConfig>,
    #[serde(rename = "Page", alias = "page")]
    pub page: Option<LlmGenerationConfig>,
}

impl Default for SegmentProcessing {
    fn default() -> Self {
        Self {
            title: Some(AutoGenerationConfig::default()),
            section_header: Some(AutoGenerationConfig::default()),
            text: Some(AutoGenerationConfig::default()),
            list_item: Some(AutoGenerationConfig::default()),
            table: Some(LlmGenerationConfig::default()),
            picture: Some(PictureGenerationConfig::default()),
            caption: Some(AutoGenerationConfig::default()),
            formula: Some(LlmGenerationConfig::default()),
            footnote: Some(AutoGenerationConfig::default()),
            page_header: Some(AutoGenerationConfig::default()),
            page_footer: Some(AutoGenerationConfig::default()),
            page: Some(LlmGenerationConfig::default()),
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
pub enum EmbedSource {
    HTML,
    Markdown,
    LLM,
    Content,
}

fn default_embed_sources() -> Vec<EmbedSource> {
    vec![EmbedSource::Markdown]
}

// TODO: Change to macro
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the processing and generation for the segment.
/// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
///   or `Auto` to only crop when needed for post-processing.
/// - `html` is the HTML output for the segment, generated either through heuristic (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through heuristic (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `embed_sources` defines which content sources will be included in the chunk's embed field and counted towards the chunk length.
///   The array's order determines the sequence in which content appears in the embed field (e.g., [Markdown, LLM] means Markdown content
///   is followed by LLM content). This directly affects what content is available for embedding and retrieval.
pub struct AutoGenerationConfig {
    #[serde(default = "default_cropping_strategy")]
    #[schema(value_type = CroppingStrategy, default = "Auto")]
    pub crop_image: CroppingStrategy,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub html: GenerationStrategy,
    pub llm: Option<String>,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub markdown: GenerationStrategy,
    #[serde(default = "default_embed_sources")]
    #[schema(value_type = Vec<EmbedSource>, default = "[Markdown]")]
    pub embed_sources: Vec<EmbedSource>,
    /// Use the full page image as context for LLM generation
    #[serde(default)]
    #[schema(default = false)]
    pub extended_context: bool,
}

fn default_cropping_strategy() -> CroppingStrategy {
    CroppingStrategy::Auto
}

fn default_picture_cropping_strategy() -> PictureCroppingStrategy {
    PictureCroppingStrategy::All
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
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the processing and generation for the segment.
/// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
///   or `Auto` to only crop when needed for post-processing.
/// - `html` is the HTML output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through huerstics (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `embed_sources` defines which content sources will be included in the chunk's embed field and counted towards the chunk length.
///   The array's order determines the sequence in which content appears in the embed field (e.g., [Markdown, LLM] means Markdown content
///   is followed by LLM content). This directly affects what content is available for embedding and retrieval.
pub struct LlmGenerationConfig {
    #[serde(default = "default_cropping_strategy")]
    #[schema(value_type = CroppingStrategy, default = "Auto")]
    pub crop_image: CroppingStrategy,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub html: GenerationStrategy,
    /// Prompt for the LLM model
    pub llm: Option<String>,
    #[serde(default = "default_llm_generation_strategy")]
    #[schema(default = "LLM")]
    pub markdown: GenerationStrategy,
    #[serde(default = "default_embed_sources")]
    #[schema(value_type = Vec<EmbedSource>, default = "[Markdown]")]
    pub embed_sources: Vec<EmbedSource>,
    /// Use the full page image as context for LLM generation
    #[serde(default = "default_extended_context")]
    #[schema(default = false)]
    pub extended_context: bool,
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::LLM,
            llm: None,
            markdown: GenerationStrategy::LLM,
            crop_image: default_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
/// Controls the processing and generation for the segment.
/// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
///   or `Auto` to only crop when needed for post-processing.
/// - `html` is the HTML output for the segment, generated either through heuristic (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
/// - `markdown` is the Markdown output for the segment, generated either through heuristic (`Auto`) or using Chunkr fine-tuned models (`LLM`)
/// - `embed_sources` defines which content sources will be included in the chunk's embed field and counted towards the chunk length.
///   The array's order determines the sequence in which content appears in the embed field (e.g., [Markdown, LLM] means Markdown content
///   is followed by LLM content). This directly affects what content is available for embedding and retrieval.
pub struct PictureGenerationConfig {
    #[serde(default = "default_picture_cropping_strategy")]
    #[schema(value_type = PictureCroppingStrategy, default = "All")]
    pub crop_image: PictureCroppingStrategy,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub html: GenerationStrategy,
    /// Prompt for the LLM model
    pub llm: Option<String>,
    #[serde(default = "default_auto_generation_strategy")]
    #[schema(default = "Auto")]
    pub markdown: GenerationStrategy,
    #[serde(default = "default_embed_sources")]
    #[schema(value_type = Vec<EmbedSource>, default = "[Markdown]")]
    pub embed_sources: Vec<EmbedSource>,
    /// Use the full page image as context for LLM generation
    #[serde(default = "default_extended_context")]
    #[schema(default = false)]
    pub extended_context: bool,
}

impl Default for PictureGenerationConfig {
    fn default() -> Self {
        Self {
            html: GenerationStrategy::Auto,
            llm: None,
            markdown: GenerationStrategy::Auto,
            crop_image: default_picture_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
        }
    }
}

fn default_extended_context() -> bool {
    false
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

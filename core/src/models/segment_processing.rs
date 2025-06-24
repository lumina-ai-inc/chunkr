use crate::models::cropping::{CroppingStrategy, PictureCroppingStrategy};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Deserializer, Serialize};
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
    pub table: Option<TableGenerationConfig>,
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
            table: Some(TableGenerationConfig::default()),
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
    Content,
    #[deprecated]
    HTML,
    #[deprecated]
    Markdown,
    LLM,
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
pub enum SegmentFormat {
    Html,
    Markdown,
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

fn default_embed_sources() -> Vec<EmbedSource> {
    vec![EmbedSource::Markdown]
}

fn default_cropping_strategy() -> CroppingStrategy {
    CroppingStrategy::Auto
}

fn default_picture_cropping_strategy() -> PictureCroppingStrategy {
    PictureCroppingStrategy::All
}

fn default_output_format() -> SegmentFormat {
    SegmentFormat::Markdown
}

fn default_table_output_format() -> SegmentFormat {
    SegmentFormat::Html
}

fn default_auto_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::Auto
}

fn default_llm_generation_strategy() -> GenerationStrategy {
    GenerationStrategy::LLM
}

fn default_extended_context() -> bool {
    false
}

/// Helper function to determine format and strategy from legacy html/markdown fields
fn resolve_format_and_strategy(
    html: Option<GenerationStrategy>,
    markdown: Option<GenerationStrategy>,
    default_format: SegmentFormat,
    default_strategy: GenerationStrategy,
) -> (SegmentFormat, GenerationStrategy) {
    match (html, markdown) {
        // Both fields set with same strategy
        (Some(GenerationStrategy::LLM), Some(GenerationStrategy::LLM)) => {
            // If both are LLM, prefer the default format for this struct type
            (default_format, GenerationStrategy::LLM)
        }
        (Some(GenerationStrategy::Auto), Some(GenerationStrategy::Auto)) => {
            // If both are Auto, prefer the default format for this struct type
            (default_format, GenerationStrategy::Auto)
        }
        // One LLM, one Auto - use the LLM one, prefer HTML format for LLM
        (Some(GenerationStrategy::LLM), Some(GenerationStrategy::Auto)) => {
            (SegmentFormat::Html, GenerationStrategy::LLM)
        }
        (Some(GenerationStrategy::Auto), Some(GenerationStrategy::LLM)) => {
            (SegmentFormat::Markdown, GenerationStrategy::LLM)
        }
        // Only HTML set
        (Some(html_strategy), None) => (SegmentFormat::Html, html_strategy),
        // Only Markdown set
        (None, Some(md_strategy)) => (SegmentFormat::Markdown, md_strategy),
        // Neither set - use defaults for this struct type
        (None, None) => (default_format, default_strategy),
    }
}

macro_rules! generation_config {
    (
        $struct_name:ident {
            crop_image: {
                type: $crop_type:ty,
                default_fn: $crop_default:literal,
                schema_default: $crop_schema_default:literal,
            },
            strategy: {
                default_fn: $strategy_default:expr,
                serde_default: $serde_default:literal,
                schema_default: $strategy_schema_default:literal,
            },
            default_format_fn: $default_format_fn:expr,
        }
    ) => {
        #[derive(Debug, Serialize, Clone, ToSchema, ToSql, FromSql)]
        /// Controls the processing and generation for the segment.
        /// - `crop_image` controls whether to crop the file's images to the segment's bounding box.
        ///   The cropped image will be stored in the segment's `image` field. Use `All` to always crop,
        ///   or `Auto` to only crop when needed for post-processing.
        /// - `format` specifies the output format: `Html` or `Markdown`
        /// - `strategy` determines how the content is generated: `Auto` (heuristics) or `LLM` (using Chunkr fine-tuned models)
        /// - `llm` is the LLM-generated output for the segment, this uses off-the-shelf models to generate a custom output for the segment
        /// - `embed_sources` defines which content sources will be included in the chunk's embed field and counted towards the chunk length.
        ///   The array's order determines the sequence in which content appears in the embed field (e.g., [Content, LLM] means content (markdown or html)
        ///   is followed by LLM content).
        ///
        /// **Deprecated fields (for backwards compatibility):**
        /// - `html` - **DEPRECATED**: Use `format: Html` and `strategy` instead
        /// - `markdown` - **DEPRECATED**: Use `format: Markdown` and `strategy` instead
        pub struct $struct_name {
            #[serde(default = $crop_default)]
            #[schema(value_type = $crop_type, default = $crop_schema_default)]
            pub crop_image: $crop_type,
            #[serde(default = "default_output_format")]
            #[schema(default = "Markdown")]
            pub format: SegmentFormat,
            #[serde(default = $serde_default)]
            #[schema(default = $strategy_schema_default)]
            pub strategy: GenerationStrategy,
            /// Prompt for the LLM model
            pub llm: Option<String>,
            #[serde(default = "default_embed_sources")]
            #[schema(value_type = Vec<EmbedSource>, default = "[Content]")]
            pub embed_sources: Vec<EmbedSource>,
            /// Use the full page image as context for LLM generation
            #[serde(default = "default_extended_context")]
            #[schema(default = false)]
            pub extended_context: bool,
            /// **DEPRECATED**: Use `format: OutputFormat::Html` and `strategy` instead.
            #[deprecated]
            #[serde(skip_serializing_if = "Option::is_none")]
            pub html: Option<GenerationStrategy>,
            /// **DEPRECATED**: Use `format: OutputFormat::Markdown` and `strategy` instead.
            #[deprecated]
            #[serde(skip_serializing_if = "Option::is_none")]
            pub markdown: Option<GenerationStrategy>,
        }

        impl<'de> Deserialize<'de> for $struct_name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                #[derive(Deserialize)]
                struct Helper {
                    #[serde(default = $crop_default)]
                    crop_image: $crop_type,
                    #[serde(default)]
                    format: Option<SegmentFormat>,
                    #[serde(default)]
                    strategy: Option<GenerationStrategy>,
                    #[serde(default)]
                    llm: Option<String>,
                    #[serde(default = "default_embed_sources")]
                    embed_sources: Vec<EmbedSource>,
                    #[serde(default = "default_extended_context")]
                    extended_context: bool,
                    // Legacy fields
                    #[serde(default)]
                    html: Option<GenerationStrategy>,
                    #[serde(default)]
                    markdown: Option<GenerationStrategy>,
                }

                let helper = Helper::deserialize(deserializer)?;

                // Get the default format and strategy for this struct type
                let default_format = $default_format_fn();
                let default_strategy = $strategy_default();

                // Determine format and strategy from new or legacy fields
                let (resolved_format, resolved_strategy) = match (helper.format, helper.strategy) {
                    // New format: both format and strategy are provided
                    (Some(format), Some(strategy)) => (format, strategy),
                    // Partial new format: only format provided, use default strategy
                    (Some(format), None) => (format, default_strategy),
                    // Partial new format: only strategy provided, use default format
                    (None, Some(strategy)) => (default_format, strategy),
                    // Legacy format: use html/markdown fields
                    (None, None) => resolve_format_and_strategy(
                        helper.html.clone(),
                        helper.markdown.clone(),
                        default_format,
                        default_strategy,
                    ),
                };

                Ok(Self {
                    crop_image: helper.crop_image,
                    format: resolved_format,
                    strategy: resolved_strategy,
                    llm: helper.llm,
                    embed_sources: helper.embed_sources,
                    extended_context: helper.extended_context,
                    html: helper.html,
                    markdown: helper.markdown,
                })
            }
        }
    };
}

generation_config! {
    AutoGenerationConfig {
        crop_image: {
            type: CroppingStrategy,
            default_fn: "default_cropping_strategy",
            schema_default: "Auto",
        },
        strategy: {
            default_fn: default_auto_generation_strategy,
            serde_default: "default_auto_generation_strategy",
            schema_default: "Auto",
        },
        default_format_fn: default_output_format,
    }
}

impl Default for AutoGenerationConfig {
    fn default() -> Self {
        Self {
            format: default_output_format(),
            strategy: default_auto_generation_strategy(),
            llm: None,
            crop_image: default_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
            html: None,
            markdown: None,
        }
    }
}

generation_config! {
    LlmGenerationConfig {
        crop_image: {
            type: CroppingStrategy,
            default_fn: "default_cropping_strategy",
            schema_default: "Auto",
        },
        strategy: {
            default_fn: default_llm_generation_strategy,
            serde_default: "default_llm_generation_strategy",
            schema_default: "LLM",
        },
        default_format_fn: default_output_format,
    }
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            format: default_output_format(),
            strategy: default_llm_generation_strategy(),
            llm: None,
            crop_image: default_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
            html: None,
            markdown: None,
        }
    }
}

generation_config! {
    PictureGenerationConfig {
        crop_image: {
            type: PictureCroppingStrategy,
            default_fn: "default_picture_cropping_strategy",
            schema_default: "All",
        },
        strategy: {
            default_fn: default_auto_generation_strategy,
            serde_default: "default_auto_generation_strategy",
            schema_default: "Auto",
        },
        default_format_fn: default_output_format,
    }
}

impl Default for PictureGenerationConfig {
    fn default() -> Self {
        Self {
            format: default_output_format(),
            strategy: default_llm_generation_strategy(),
            llm: None,
            crop_image: default_picture_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
            html: None,
            markdown: None,
        }
    }
}

generation_config! {
    TableGenerationConfig {
        crop_image: {
            type: CroppingStrategy,
            default_fn: "default_cropping_strategy",
            schema_default: "Auto",
        },
        strategy: {
            default_fn: default_llm_generation_strategy,
            serde_default: "default_llm_generation_strategy",
            schema_default: "LLM",
        },
        default_format_fn: default_table_output_format,
    }
}

impl Default for TableGenerationConfig {
    fn default() -> Self {
        Self {
            format: default_table_output_format(),
            strategy: default_llm_generation_strategy(),
            llm: None,
            crop_image: default_cropping_strategy(),
            embed_sources: default_embed_sources(),
            extended_context: default_extended_context(),
            html: None,
            markdown: None,
        }
    }
}

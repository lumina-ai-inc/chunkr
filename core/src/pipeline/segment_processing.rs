use crate::configs::llm_config::create_messages_from_template;
use crate::models::output::{Segment, SegmentType};
use crate::models::pipeline::Pipeline;
use crate::models::segment_processing::{
    AutoGenerationConfig, GenerationStrategy, LlmGenerationConfig, PictureGenerationConfig,
};
use crate::models::task::{Configuration, Status};
use crate::models::upload::ErrorHandlingStrategy;
use crate::utils::services::file_operations::get_file_url;
use crate::utils::services::{html, llm, markdown};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::NamedTempFile;

lazy_static! {
    static ref NUMBERED_LIST_REGEX: Regex = Regex::new(r"^(\d+)\.\s+(.+)$").unwrap();
}

/// Parameters for LLM content generation
#[derive(Clone)]
struct LlmGenerationParams<'a> {
    segment_id: &'a str,
    image_folder_location: &'a str,
    segment_image: Arc<NamedTempFile>,
    page_image: Option<Arc<NamedTempFile>>,
    extended_context: bool,
    llm_fallback_content: Option<String>,
    configuration: &'a Configuration,
}

/// Parameters for HTML/Markdown generation
struct ContentGenerationParams<'a> {
    segment: &'a Segment,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &'a GenerationStrategy,
    extended_context: bool,
    fallback_content: Option<String>,
    image_folder_location: &'a str,
    configuration: &'a Configuration,
}

/// Parameters for generation strategy application
struct StrategyParams<'a, T: ContentGenerator> {
    segment_id: &'a str,
    image_folder_location: &'a str,
    generator: &'a T,
    auto_content: &'a str,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &'a GenerationStrategy,
    extended_context: bool,
    override_auto: String,
    llm_fallback_content: Option<String>,
    configuration: &'a Configuration,
}

/// Parameters for standalone LLM generation
struct StandaloneLlmParams<'a> {
    segment: &'a Segment,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    llm_prompt: Option<String>,
    extended_context: bool,
    llm_fallback_content: Option<String>,
    image_folder_location: &'a str,
    configuration: &'a Configuration,
}

impl<'a> LlmGenerationParams<'a> {
    fn from_strategy_params<T: ContentGenerator>(
        params: &'a StrategyParams<'a, T>,
        segment_image: Arc<NamedTempFile>,
    ) -> Self {
        Self {
            segment_id: params.segment_id,
            image_folder_location: params.image_folder_location,
            segment_image,
            page_image: params.page_image.clone(),
            extended_context: params.extended_context,
            llm_fallback_content: params.llm_fallback_content.clone(),
            configuration: params.configuration,
        }
    }
}

impl<'a, T: ContentGenerator> StrategyParams<'a, T> {
    fn from_content_params(
        params: &'a ContentGenerationParams<'a>,
        generator: &'a T,
        override_auto: String,
    ) -> Self {
        Self {
            segment_id: &params.segment.segment_id,
            image_folder_location: params.image_folder_location,
            generator,
            auto_content: &params.segment.content,
            segment_image: params.segment_image.clone(),
            page_image: params.page_image.clone(),
            generation_strategy: params.generation_strategy,
            extended_context: params.extended_context
                && params.generation_strategy == &GenerationStrategy::LLM,
            override_auto,
            llm_fallback_content: params.fallback_content.clone(),
            configuration: params.configuration,
        }
    }
}

impl<'a> ContentGenerationParams<'a> {
    fn new(
        segment: &'a Segment,
        segment_image: Option<Arc<NamedTempFile>>,
        page_image: Option<Arc<NamedTempFile>>,
        generation_strategy: &'a GenerationStrategy,
        extended_context: bool,
        fallback_content: Option<String>,
        image_folder_location: &'a str,
        configuration: &'a Configuration,
    ) -> Self {
        Self {
            segment,
            segment_image,
            page_image,
            generation_strategy,
            extended_context,
            fallback_content,
            image_folder_location,
            configuration,
        }
    }
}

impl<'a> StandaloneLlmParams<'a> {
    fn new(
        segment: &'a Segment,
        segment_image: Option<Arc<NamedTempFile>>,
        page_image: Option<Arc<NamedTempFile>>,
        llm_prompt: Option<String>,
        extended_context: bool,
        llm_fallback_content: Option<String>,
        image_folder_location: &'a str,
        configuration: &'a Configuration,
    ) -> Self {
        Self {
            segment,
            segment_image,
            page_image,
            llm_prompt,
            extended_context,
            llm_fallback_content,
            image_folder_location,
            configuration,
        }
    }
}

trait ContentGenerator {
    fn clean_list_item(content: &str) -> String {
        content
            .trim_start_matches(&['-', '*', '•', '●', ' ', ''][..])
            .trim()
            .to_string()
    }
    fn generate_auto(&self, content: &str) -> String;
    fn template_key(&self, extended_context: bool) -> &'static str;
    fn segment_type(&self) -> SegmentType;

    async fn process_llm(
        &self,
        params: &LlmGenerationParams<'_>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut values = HashMap::new();

        // Keep segment image as is - it's already a reasonable size
        let segment_image_url = get_file_url(
            &params.segment_image,
            &format!("{}/{}.jpg", params.image_folder_location, params.segment_id),
        )
        .await?;
        values.insert("image_url".to_string(), segment_image_url);

        if params.extended_context {
            if let Some(page_img) = &params.page_image {
                // Use the page image as is
                let page_image_url = get_file_url(
                    page_img,
                    &format!(
                        "{}/{}_page.jpg",
                        params.image_folder_location, params.segment_id
                    ),
                )
                .await?;
                values.insert("page_image_url".to_string(), page_image_url);
            } else {
                return Err("Page image not found".into());
            }
        }

        let template_key = self.template_key(params.extended_context);
        let messages = create_messages_from_template(template_key, &values)?;

        let fence_type = match (template_key, self.segment_type()) {
            (_, SegmentType::Formula) => Some("latex"),
            (key, _) if key.starts_with("md_") => Some("markdown"),
            _ => Some("html"),
        };

        llm::try_extract_from_llm(
            messages,
            fence_type,
            params.llm_fallback_content.clone(),
            params.configuration.llm_processing.clone(),
        )
        .await
    }

    async fn generate_llm(
        &self,
        params: &LlmGenerationParams<'_>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

struct HtmlGenerator {
    segment_type: SegmentType,
}

impl ContentGenerator for HtmlGenerator {
    fn generate_auto(&self, content: &str) -> String {
        match self.segment_type {
            SegmentType::Caption => format!("<span class=\"caption\">{}</span>", content),
            SegmentType::Footnote => format!("<span class=\"footnote\">{}</span>", content),
            SegmentType::Formula => format!("<span class=\"formula\">{}</span>", content),
            SegmentType::ListItem => {
                if let Some(captures) = NUMBERED_LIST_REGEX.captures(content.trim()) {
                    let start_number = captures.get(1).unwrap().as_str().parse::<i32>().unwrap();
                    let item = captures.get(2).unwrap().as_str();
                    format!("<ol start=\"{}\"><li>{}</li></ol>", start_number, item)
                } else {
                    format!("<ul><li>{}</li></ul>", Self::clean_list_item(content))
                }
            }
            SegmentType::Page => format!("<div class=\"page\">{}</div>", content),
            SegmentType::PageFooter => format!("<div class=\"page-footer\">{}</div>", content),
            SegmentType::PageHeader => format!("<div class=\"page-header\">{}</div>", content),
            SegmentType::Picture => format!("<img src='' alt='{}' />", content),
            SegmentType::SectionHeader => format!("<h2>{}</h2>", content),
            SegmentType::Table => format!("<table><tr><td>{}</td></tr></table>", content),
            SegmentType::Text => format!("<p>{}</p>", content),
            SegmentType::Title => format!("<h1>{}</h1>", content),
        }
    }

    fn template_key(&self, extended_context: bool) -> &'static str {
        match self.segment_type.clone() {
            SegmentType::Table => {
                if extended_context {
                    "html_table_extended"
                } else {
                    "html_table"
                }
            }
            SegmentType::Picture => {
                if extended_context {
                    "html_picture_extended"
                } else {
                    "html_picture"
                }
            }
            SegmentType::Formula => {
                if extended_context {
                    "formula_extended"
                } else {
                    "formula"
                }
            }
            SegmentType::Page => "html_page",
            SegmentType::Caption => {
                if extended_context {
                    "html_caption_extended"
                } else {
                    "html_caption"
                }
            }
            SegmentType::Footnote => {
                if extended_context {
                    "html_footnote_extended"
                } else {
                    "html_footnote"
                }
            }
            SegmentType::ListItem => {
                if extended_context {
                    "html_list_item_extended"
                } else {
                    "html_list_item"
                }
            }
            SegmentType::PageFooter => {
                if extended_context {
                    "html_page_footer_extended"
                } else {
                    "html_page_footer"
                }
            }
            SegmentType::PageHeader => {
                if extended_context {
                    "html_page_header_extended"
                } else {
                    "html_page_header"
                }
            }
            SegmentType::SectionHeader => {
                if extended_context {
                    "html_section_header_extended"
                } else {
                    "html_section_header"
                }
            }
            SegmentType::Text => {
                if extended_context {
                    "html_text_extended"
                } else {
                    "html_text"
                }
            }
            SegmentType::Title => {
                if extended_context {
                    "html_title_extended"
                } else {
                    "html_title"
                }
            }
        }
    }

    fn segment_type(&self) -> SegmentType {
        self.segment_type.clone()
    }

    async fn generate_llm(
        &self,
        params: &LlmGenerationParams<'_>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self.process_llm(params).await?;

        if self.segment_type() == SegmentType::Formula {
            Ok(format!("<span class=\"formula\">{}</span>", content))
        } else {
            Ok(content)
        }
    }
}

struct MarkdownGenerator {
    segment_type: SegmentType,
}

impl ContentGenerator for MarkdownGenerator {
    fn generate_auto(&self, content: &str) -> String {
        match self.segment_type {
            SegmentType::Caption => format!("_{}_", content),
            SegmentType::Footnote => format!("[^{}]", content),
            SegmentType::Formula => format!("${}$", content),
            SegmentType::ListItem => {
                if let Some(captures) = NUMBERED_LIST_REGEX.captures(content.trim()) {
                    format!(
                        "{}. {}",
                        captures.get(1).unwrap().as_str(),
                        captures.get(2).unwrap().as_str()
                    )
                } else {
                    format!("- {}", Self::clean_list_item(content))
                }
            }
            SegmentType::Page => content.to_string(),
            SegmentType::PageFooter | SegmentType::PageHeader => content.to_string(),
            SegmentType::Picture => format!("![{}]()", content),
            SegmentType::SectionHeader => format!("## {}", content),
            SegmentType::Table => format!("| {} |", content),
            SegmentType::Text => content.to_string(),
            SegmentType::Title => format!("# {}", content),
        }
    }

    fn template_key(&self, extended_context: bool) -> &'static str {
        match self.segment_type.clone() {
            SegmentType::Table => {
                if extended_context {
                    "md_table_extended"
                } else {
                    "md_table"
                }
            }
            SegmentType::Picture => {
                if extended_context {
                    "md_picture_extended"
                } else {
                    "md_picture"
                }
            }
            SegmentType::Formula => {
                if extended_context {
                    "formula_extended"
                } else {
                    "formula"
                }
            }
            SegmentType::Page => "md_page",

            SegmentType::Caption => {
                if extended_context {
                    "md_caption_extended"
                } else {
                    "md_caption"
                }
            }
            SegmentType::Footnote => {
                if extended_context {
                    "md_footnote_extended"
                } else {
                    "md_footnote"
                }
            }
            SegmentType::ListItem => {
                if extended_context {
                    "md_list_item_extended"
                } else {
                    "md_list_item"
                }
            }
            SegmentType::PageFooter => {
                if extended_context {
                    "md_page_footer_extended"
                } else {
                    "md_page_footer"
                }
            }
            SegmentType::PageHeader => {
                if extended_context {
                    "md_page_header_extended"
                } else {
                    "md_page_header"
                }
            }
            SegmentType::SectionHeader => {
                if extended_context {
                    "md_section_header_extended"
                } else {
                    "md_section_header"
                }
            }
            SegmentType::Text => {
                if extended_context {
                    "md_text_extended"
                } else {
                    "md_text"
                }
            }
            SegmentType::Title => {
                if extended_context {
                    "md_title_extended"
                } else {
                    "md_title"
                }
            }
        }
    }

    fn segment_type(&self) -> SegmentType {
        self.segment_type.clone()
    }

    async fn generate_llm(
        &self,
        params: &LlmGenerationParams<'_>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self.process_llm(params).await?;

        if self.segment_type() == SegmentType::Formula {
            Ok(format!("${content}$"))
        } else {
            Ok(content)
        }
    }
}

fn convert_checkboxes(content: &str) -> String {
    content
        .replace(":selected:", "☑")
        .replace(":unselected:", "☐")
}

fn convert_checkboxes_html(content: &str) -> String {
    content
        .replace(":selected:", "<input type=\"checkbox\" checked>")
        .replace(":unselected:", "<input type=\"checkbox\">")
}

fn convert_checkboxes_markdown(content: &str) -> String {
    content
        .replace(":selected:", "[x]")
        .replace(":unselected:", "[ ]")
}

async fn apply_generation_strategy<T: ContentGenerator>(
    params: &StrategyParams<'_, T>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if !params.override_auto.is_empty() && params.generation_strategy == &GenerationStrategy::Auto {
        return Ok(params.override_auto.clone());
    }

    if params.segment_image.is_none() {
        // Cannot use LLM without segment image, fallback to auto
        return Ok(params.generator.generate_auto(params.auto_content));
    }
    let segment_image = params.segment_image.clone().unwrap(); // Safe unwrap due to check above

    match params.generation_strategy {
        GenerationStrategy::LLM => {
            let llm_params = LlmGenerationParams::from_strategy_params(params, segment_image);
            Ok(params.generator.generate_llm(&llm_params).await?)
        }
        GenerationStrategy::Auto => Ok(params.generator.generate_auto(params.auto_content)),
    }
}

async fn generate_html(
    params: &ContentGenerationParams<'_>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = HtmlGenerator {
        segment_type: params.segment.segment_type.clone(),
    };

    let strategy_params =
        StrategyParams::from_content_params(params, &generator, params.segment.html.clone());

    Ok(html::clean_img_tags(
        &apply_generation_strategy(&strategy_params).await?,
    ))
}

async fn generate_markdown(
    params: &ContentGenerationParams<'_>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = MarkdownGenerator {
        segment_type: params.segment.segment_type.clone(),
    };

    let strategy_params =
        StrategyParams::from_content_params(params, &generator, params.segment.markdown.clone());

    Ok(markdown::clean_img_tags(
        &apply_generation_strategy(&strategy_params).await?,
    ))
}

async fn generate_llm(
    params: &StandaloneLlmParams<'_>,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    if params.llm_prompt.is_none() || params.segment_image.is_none() {
        return Ok(None);
    }
    let segment_image = params.segment_image.clone().unwrap(); // Safe unwrap

    let mut values = HashMap::new();
    let segment_image_url = get_file_url(
        &segment_image,
        &format!(
            "{}/{}.jpg",
            params.image_folder_location, params.segment.segment_id
        ),
    )
    .await?;
    values.insert(
        "segment_type".to_string(),
        params.segment.segment_type.to_string(),
    );
    values.insert(
        "user_prompt".to_string(),
        params.llm_prompt.clone().unwrap(),
    );
    values.insert("image_url".to_string(), segment_image_url);

    let template_key = if params.extended_context {
        if let Some(page_img) = &params.page_image {
            // Use the page image as is
            let page_image_url = get_file_url(
                page_img,
                &format!(
                    "{}/{}_page.jpg",
                    params.image_folder_location, params.segment.segment_id
                ),
            )
            .await?;
            values.insert("page_image_url".to_string(), page_image_url);
            "llm_segment_extended"
        } else {
            "llm_segment"
        }
    } else {
        "llm_segment"
    };

    let messages = create_messages_from_template(template_key, &values)?;
    let result = llm::try_extract_from_llm(
        messages,
        None, // LLM field extraction doesn't assume a fence type by default
        params.llm_fallback_content.clone(),
        params.configuration.llm_processing.clone(),
    )
    .await?;

    Ok(Some(result))
}

async fn process_segment(
    segment: &mut Segment,
    configuration: &Configuration,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    image_folder_location: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (html_strategy, markdown_strategy, llm_prompt, extended_context) = match segment
        .segment_type
        .clone()
    {
        SegmentType::Table | SegmentType::Formula | SegmentType::Page => {
            let config: &LlmGenerationConfig = match segment.segment_type {
                SegmentType::Table => configuration.segment_processing.table.as_ref().unwrap(),
                SegmentType::Formula => configuration.segment_processing.formula.as_ref().unwrap(),
                SegmentType::Page => configuration.segment_processing.page.as_ref().unwrap(),
                _ => unreachable!(),
            };
            (
                &config.html,
                &config.markdown,
                &config.llm,
                config.extended_context,
            )
        }
        SegmentType::Picture => {
            let config: &PictureGenerationConfig =
                configuration.segment_processing.picture.as_ref().unwrap();
            (
                &config.html,
                &config.markdown,
                &config.llm,
                config.extended_context,
            )
        }
        segment_type => {
            let config: &AutoGenerationConfig = match segment_type {
                SegmentType::Title => configuration.segment_processing.title.as_ref().unwrap(),
                SegmentType::SectionHeader => configuration
                    .segment_processing
                    .section_header
                    .as_ref()
                    .unwrap(),
                SegmentType::Text => configuration.segment_processing.text.as_ref().unwrap(),
                SegmentType::ListItem => {
                    configuration.segment_processing.list_item.as_ref().unwrap()
                }
                SegmentType::Caption => configuration.segment_processing.caption.as_ref().unwrap(),
                SegmentType::Footnote => {
                    configuration.segment_processing.footnote.as_ref().unwrap()
                }
                SegmentType::PageHeader => configuration
                    .segment_processing
                    .page_header
                    .as_ref()
                    .unwrap(),
                SegmentType::PageFooter => configuration
                    .segment_processing
                    .page_footer
                    .as_ref()
                    .unwrap(),
                _ => unreachable!(),
            };
            (
                &config.html,
                &config.markdown,
                &config.llm,
                config.extended_context,
            )
        }
    };

    let (fallback_html, fallback_markdown, fallback_llm) = match segment.segment_type.clone() {
        SegmentType::Table => (
            Some(segment.html.clone()).filter(|s| !s.is_empty()),
            Some(segment.markdown.clone()).filter(|s| !s.is_empty()),
            None,
        ),
        _ => (None, None, None),
    };

    // Process HTML with error handling using new parameter struct
    let html_params = ContentGenerationParams::new(
        segment,
        segment_image.clone(),
        page_image.clone(),
        html_strategy,
        extended_context,
        fallback_html,
        image_folder_location,
        configuration,
    );

    // Process Markdown with error handling using new parameter struct
    let markdown_params = ContentGenerationParams::new(
        segment,
        segment_image.clone(),
        page_image.clone(),
        markdown_strategy,
        extended_context,
        fallback_markdown,
        image_folder_location,
        configuration,
    );

    // Process LLM with error handling using new parameter struct
    let llm_params = StandaloneLlmParams::new(
        segment,
        segment_image,
        page_image,
        llm_prompt.clone(),
        extended_context,
        fallback_llm,
        image_folder_location,
        configuration,
    );

    // Create futures for all three operations so they can run concurrently
    let html_future = async {
        match generate_html(&html_params).await {
            Ok(content) => Ok(content),
            Err(e) => {
                if configuration.error_handling == ErrorHandlingStrategy::Continue {
                    let html_generator = HtmlGenerator {
                        segment_type: segment.segment_type.clone(),
                    };
                    Ok(html_generator.generate_auto(&segment.content))
                } else {
                    Err(e)
                }
            }
        }
    };

    let markdown_future = async {
        match generate_markdown(&markdown_params).await {
            Ok(content) => Ok(content),
            Err(e) => {
                if configuration.error_handling == ErrorHandlingStrategy::Continue {
                    let markdown_generator = MarkdownGenerator {
                        segment_type: segment.segment_type.clone(),
                    };
                    Ok(markdown_generator.generate_auto(&segment.content))
                } else {
                    Err(e)
                }
            }
        }
    };

    let llm_future = async {
        match generate_llm(&llm_params).await {
            Ok(content) => Ok(content),
            Err(e) => {
                if configuration.error_handling == ErrorHandlingStrategy::Continue {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    };

    let (html, markdown, llm) = tokio::try_join!(html_future, markdown_future, llm_future)?;

    segment.content = convert_checkboxes(&segment.content);
    segment.html = convert_checkboxes_html(&html);
    segment.markdown = convert_checkboxes_markdown(&markdown);
    segment.llm = llm;
    Ok(())
}

/// Process the segments and creates the html, llm and markdown fields
///
/// This function will generate the html, llm and markdown fields for all the segments in parallel.
/// Depending on the configuration, each segment will either be processed using heuristic or by a LLM.
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let task = pipeline.get_task()?;
    let configuration = task.configuration.clone();
    let segment_images = pipeline.segment_images.clone();

    // Simply clone out the Option<Vec<…>> and default to an empty Vec if missing
    let page_images: Vec<_> = pipeline.page_images.clone().unwrap_or_default();

    let futures: Vec<_> = pipeline
        .chunks
        .iter_mut()
        .flat_map(|chunk| {
            chunk.segments.iter_mut().map(|segment| {
                let page_index = if segment.page_number > 0 {
                    (segment.page_number - 1) as usize
                } else {
                    0
                };
                let segment_page_image = page_images.get(page_index).cloned();
                let segment_image_ref = segment_images.get(&segment.segment_id);
                let segment_image_cloned = segment_image_ref.map(|r| r.value().clone());

                process_segment(
                    segment,
                    &configuration,
                    segment_image_cloned,
                    segment_page_image,
                    &task.image_folder_location,
                )
            })
        })
        .collect();

    println!("Processing {:?} segments concurrently", futures.len());
    match futures::future::try_join_all(futures).await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error processing segments: {:?}", e);
            return Err(e.to_string().into());
        }
    }

    Ok(())
}

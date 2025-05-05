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

trait ContentGenerator {
    fn clean_list_item(content: &str) -> String {
        content
            .trim_start_matches(&['-', '*', '•', '●', ' ', ''][..])
            .trim()
            .to_string()
    }
    fn generate_auto(&self, content: &str) -> String;
    fn template_key(&self, use_extended_context: bool) -> &'static str;
    fn segment_type(&self) -> SegmentType;
    async fn process_llm(
        &self,
        segment_id: &str,
        image_folder_location: &str,
        segment_image: Arc<NamedTempFile>,
        page_image: Option<Arc<NamedTempFile>>,
        use_extended_context: bool,
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut values = HashMap::new();

        // Keep segment image as is - it's already a reasonable size
        let segment_image_url = get_file_url(
            &segment_image,
            &format!("{}/{}.jpg", image_folder_location, segment_id),
        )
        .await?;
        values.insert("image_url".to_string(), segment_image_url);

        if use_extended_context {
            if let Some(page_img) = page_image {
                // Use the page image as is
                let page_image_url = get_file_url(
                    &page_img,
                    &format!("{}/{}_page.jpg", image_folder_location, segment_id),
                )
                .await?;
                values.insert("page_image_url".to_string(), page_image_url);
            } else {
                println!(
                    "Warning: Extended context requested for segment {} but page image not found.",
                    segment_id
                );
            }
        }

        let template_key = self.template_key(use_extended_context);
        let messages = create_messages_from_template(template_key, &values)?;

        let fence_type = match (template_key, self.segment_type()) {
            (_, SegmentType::Formula) => Some("latex"),
            (key, _) if key.starts_with("md_") => Some("markdown"),
            _ => Some("html"),
        };

        llm::try_extract_from_llm(
            messages,
            fence_type,
            llm_fallback_content,
            configuration.llm_processing.clone(),
        )
        .await
    }
    async fn generate_llm(
        &self,
        segment_id: &str,
        image_folder_location: &str,
        segment_image: Arc<NamedTempFile>,
        page_image: Option<Arc<NamedTempFile>>,
        use_extended_context: bool,
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
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

    fn template_key(&self, use_extended_context: bool) -> &'static str {
        match (self.segment_type.clone(), use_extended_context) {
            (SegmentType::Table | SegmentType::Picture, true) => {
                println!(
                    "Using HTML extended context for segment_type={:?}",
                    self.segment_type
                );
                match self.segment_type {
                    SegmentType::Table => "html_table_extended",
                    SegmentType::Picture => "html_picture_extended",
                    _ => unreachable!(),
                }
            }
            (SegmentType::Page, _) => "html_page",
            (segment_type, true) => match segment_type {
                SegmentType::Caption => "html_caption_extended",
                SegmentType::Footnote => "html_footnote_extended",
                SegmentType::Formula => "formula_extended",
                SegmentType::ListItem => "html_list_item_extended",
                SegmentType::PageFooter => "html_page_footer_extended",
                SegmentType::PageHeader => "html_page_header_extended",
                SegmentType::SectionHeader => "html_section_header_extended",
                SegmentType::Text => "html_text_extended",
                SegmentType::Title => "html_title_extended",
                _ => unreachable!(),
            },
            (segment_type, false) => match segment_type {
                SegmentType::Caption => "html_caption",
                SegmentType::Footnote => "html_footnote",
                SegmentType::Formula => "formula_extended",
                SegmentType::ListItem => "html_list_item",
                SegmentType::PageFooter => "html_page_footer",
                SegmentType::PageHeader => "html_page_header",
                SegmentType::Picture => "html_picture",
                SegmentType::SectionHeader => "html_section_header",
                SegmentType::Table => "html_table",
                SegmentType::Text => "html_text",
                SegmentType::Title => "html_title",
                _ => unreachable!(),
            },
        }
    }

    fn segment_type(&self) -> SegmentType {
        self.segment_type.clone()
    }

    async fn generate_llm(
        &self,
        segment_id: &str,
        image_folder_location: &str,
        segment_image: Arc<NamedTempFile>,
        page_image: Option<Arc<NamedTempFile>>,
        use_extended_context: bool,
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self
            .process_llm(
                segment_id,
                image_folder_location,
                segment_image,
                page_image,
                use_extended_context,
                llm_fallback_content,
                configuration,
            )
            .await?;

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

    fn template_key(&self, use_extended_context: bool) -> &'static str {
        let (segment_type, result) = match (self.segment_type.clone(), use_extended_context) {
            (SegmentType::Table | SegmentType::Picture, true) => {
                println!(
                    "Using Markdown extended context for segment_type={:?}",
                    self.segment_type
                );
                (self.segment_type.clone(), true)
            }
            (segment_type, _) => (segment_type, false),
        };

        match (segment_type, result) {
            (SegmentType::Caption, false) => "md_caption",
            (SegmentType::Caption, true) => "md_caption_extended",
            (SegmentType::Footnote, false) => "md_footnote",
            (SegmentType::Footnote, true) => "md_footnote_extended",
            (SegmentType::Formula, _) => "formula_extended",
            (SegmentType::ListItem, false) => "md_list_item",
            (SegmentType::ListItem, true) => "md_list_item_extended",
            (SegmentType::Page, _) => "md_page",
            (SegmentType::PageFooter, false) => "md_page_footer",
            (SegmentType::PageFooter, true) => "md_page_footer_extended",
            (SegmentType::PageHeader, false) => "md_page_header",
            (SegmentType::PageHeader, true) => "md_page_header_extended",
            (SegmentType::Picture, false) => "md_picture",
            (SegmentType::Picture, true) => "md_picture_extended",
            (SegmentType::SectionHeader, false) => "md_section_header",
            (SegmentType::SectionHeader, true) => "md_section_header_extended",
            (SegmentType::Table, false) => "md_table",
            (SegmentType::Table, true) => "md_table_extended",
            (SegmentType::Text, false) => "md_text",
            (SegmentType::Text, true) => "md_text_extended",
            (SegmentType::Title, false) => "md_title",
            (SegmentType::Title, true) => "md_title_extended",
        }
    }

    fn segment_type(&self) -> SegmentType {
        self.segment_type.clone()
    }

    async fn generate_llm(
        &self,
        segment_id: &str,
        image_folder_location: &str,
        segment_image: Arc<NamedTempFile>,
        page_image: Option<Arc<NamedTempFile>>,
        use_extended_context: bool,
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self
            .process_llm(
                segment_id,
                image_folder_location,
                segment_image,
                page_image,
                use_extended_context,
                llm_fallback_content,
                configuration,
            )
            .await?;

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
    segment_id: &str,
    image_folder_location: &str,
    generator: &T,
    auto_content: &str,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
    use_extended_context: bool,
    override_auto: String,
    llm_fallback_content: Option<String>,
    configuration: &Configuration,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if !override_auto.is_empty() && generation_strategy == &GenerationStrategy::Auto {
        return Ok(override_auto);
    }

    if segment_image.is_none() {
        // Cannot use LLM without segment image, fallback to auto
        return Ok(generator.generate_auto(auto_content));
    }
    let segment_image = segment_image.unwrap(); // Safe unwrap due to check above

    match generation_strategy {
        GenerationStrategy::LLM => Ok(generator
            .generate_llm(
                segment_id,
                image_folder_location,
                segment_image,
                page_image,
                use_extended_context,
                llm_fallback_content,
                configuration,
            )
            .await?),
        GenerationStrategy::Auto => Ok(generator.generate_auto(auto_content)),
    }
}

async fn generate_html(
    segment: &Segment,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
    use_extended_context: bool,
    fallback_content: Option<String>,
    image_folder_location: &str,
    configuration: &Configuration,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = HtmlGenerator {
        segment_type: segment.segment_type.clone(),
    };
    Ok(html::clean_img_tags(
        &apply_generation_strategy(
            &segment.segment_id,
            image_folder_location,
            &generator,
            &segment.content.clone(),
            segment_image,
            page_image,
            generation_strategy,
            use_extended_context && generation_strategy == &GenerationStrategy::LLM, // Only pass true if LLM is used
            segment.html.clone(),
            fallback_content,
            configuration,
        )
        .await?,
    ))
}

async fn generate_markdown(
    segment: &Segment,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
    use_extended_context: bool,
    fallback_content: Option<String>,
    image_folder_location: &str,
    configuration: &Configuration,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = MarkdownGenerator {
        segment_type: segment.segment_type.clone(),
    };
    Ok(markdown::clean_img_tags(
        &apply_generation_strategy(
            &segment.segment_id,
            image_folder_location,
            &generator,
            &segment.content.clone(),
            segment_image,
            page_image,
            generation_strategy,
            use_extended_context && generation_strategy == &GenerationStrategy::LLM, // Only pass true if LLM is used
            segment.markdown.clone(),
            fallback_content,
            configuration,
        )
        .await?,
    ))
}

async fn generate_llm(
    segment: &Segment,
    segment_image: Option<Arc<NamedTempFile>>,
    page_image: Option<Arc<NamedTempFile>>,
    llm_prompt: Option<String>,
    use_extended_context: bool,
    llm_fallback_content: Option<String>,
    image_folder_location: &str,
    configuration: &Configuration,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    if llm_prompt.is_none() || segment_image.is_none() {
        return Ok(None);
    }
    let segment_image = segment_image.unwrap(); // Safe unwrap

    let mut values = HashMap::new();
    let segment_image_url = get_file_url(
        &segment_image,
        &format!("{}/{}.jpg", image_folder_location, segment.segment_id),
    )
    .await?;
    values.insert("segment_type".to_string(), segment.segment_type.to_string());
    values.insert("user_prompt".to_string(), llm_prompt.unwrap());
    values.insert("image_url".to_string(), segment_image_url);

    let template_key = if use_extended_context {
        if let Some(page_img) = page_image {
            // Use the page image as is
            let page_image_url = get_file_url(
                &page_img,
                &format!("{}/{}_page.jpg", image_folder_location, segment.segment_id),
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
        llm_fallback_content,
        configuration.llm_processing.clone(),
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
    let (html_strategy, markdown_strategy, llm_prompt, use_extended_context_config) = match segment
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
                config.use_extended_context,
            )
        }
        SegmentType::Picture => {
            let config: &PictureGenerationConfig =
                configuration.segment_processing.picture.as_ref().unwrap();
            (
                &config.html,
                &config.markdown,
                &config.llm,
                config.use_extended_context,
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
                config.use_extended_context,
            )
        }
    };

    // Determine effective use_extended_context based on config AND page_image presence
    let use_extended_context = use_extended_context_config && page_image.is_some();
    if use_extended_context_config && page_image.is_none() {
        println!(
            "Warning: Extended context requested for segment {} but page image not found. Falling back.",
            segment.segment_id
        );
    }

    let (fallback_html, fallback_markdown, fallback_llm) = match segment.segment_type.clone() {
        SegmentType::Table => (
            Some(segment.html.clone()).filter(|s| !s.is_empty()),
            Some(segment.markdown.clone()).filter(|s| !s.is_empty()),
            None,
        ),
        _ => (None, None, None),
    };

    // Process HTML with error handling
    let html = match generate_html(
        segment,
        segment_image.clone(),
        page_image.clone(),
        html_strategy,
        use_extended_context,
        fallback_html,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if configuration.error_handling == ErrorHandlingStrategy::Continue {
                let html_generator = HtmlGenerator {
                    segment_type: segment.segment_type.clone(),
                };
                html_generator.generate_auto(&segment.content)
            } else {
                return Err(e);
            }
        }
    };

    // Process Markdown with error handling
    let markdown = match generate_markdown(
        segment,
        segment_image.clone(),
        page_image.clone(),
        markdown_strategy,
        use_extended_context,
        fallback_markdown,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if configuration.error_handling == ErrorHandlingStrategy::Continue {
                let markdown_generator = MarkdownGenerator {
                    segment_type: segment.segment_type.clone(),
                };
                markdown_generator.generate_auto(&segment.content)
            } else {
                return Err(e);
            }
        }
    };

    // Process LLM with error handling
    let llm = match generate_llm(
        segment,
        segment_image.clone(),
        page_image.clone(),
        llm_prompt.clone(),
        use_extended_context,
        fallback_llm,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if configuration.error_handling == ErrorHandlingStrategy::Continue {
                None
            } else {
                return Err(e);
            }
        }
    };

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
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Processing segments".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;
    let configuration = pipeline.get_task()?.configuration.clone();
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

                // Handle potential missing page image
                if segment_page_image.is_none() {
                    println!(
                        "Warning: Page image not found for segment {} on page {}",
                        segment.segment_id, segment.page_number
                    );
                }

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

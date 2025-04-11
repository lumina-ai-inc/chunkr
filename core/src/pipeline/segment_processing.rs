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
    fn template_key(&self) -> &'static str;
    fn segment_type(&self) -> SegmentType;
    async fn process_llm(
        &self,
        segment_id: &str,
        image_folder_location: &str,
        segment_image: Arc<NamedTempFile>,
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut values = HashMap::new();
        let file_url = get_file_url(
            &segment_image,
            &format!("{}/{}.jpg", image_folder_location, segment_id),
        )
        .await?;
        values.insert("image_url".to_string(), file_url);
        let messages = create_messages_from_template(self.template_key(), &values)?;

        let fence_type = match (self.template_key(), self.segment_type()) {
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

    fn template_key(&self) -> &'static str {
        match self.segment_type {
            SegmentType::Caption => "html_caption",
            SegmentType::Footnote => "html_footnote",
            SegmentType::Formula => "formula",
            SegmentType::ListItem => "html_list_item",
            SegmentType::Page => "html_page",
            SegmentType::PageFooter => "html_page_footer",
            SegmentType::PageHeader => "html_page_header",
            SegmentType::Picture => "html_picture",
            SegmentType::SectionHeader => "html_section_header",
            SegmentType::Table => "html_table",
            SegmentType::Text => "html_text",
            SegmentType::Title => "html_title",
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
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self
            .process_llm(
                segment_id,
                image_folder_location,
                segment_image,
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

    fn template_key(&self) -> &'static str {
        match self.segment_type {
            SegmentType::Caption => "md_caption",
            SegmentType::Footnote => "md_footnote",
            SegmentType::Formula => "formula",
            SegmentType::ListItem => "md_list_item",
            SegmentType::Page => "md_page",
            SegmentType::PageFooter => "md_page_footer",
            SegmentType::PageHeader => "md_page_header",
            SegmentType::Picture => "md_picture",
            SegmentType::SectionHeader => "md_section_header",
            SegmentType::Table => "md_table",
            SegmentType::Text => "md_text",
            SegmentType::Title => "md_title",
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
        llm_fallback_content: Option<String>,
        configuration: &Configuration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self
            .process_llm(
                segment_id,
                image_folder_location,
                segment_image,
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
    generation_strategy: &GenerationStrategy,
    override_auto: String,
    llm_fallback_content: Option<String>,
    configuration: &Configuration,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if !override_auto.is_empty() && generation_strategy == &GenerationStrategy::Auto {
        return Ok(override_auto);
    }

    if segment_image.is_none() {
        return Ok(generator.generate_auto(auto_content));
    }

    match generation_strategy {
        GenerationStrategy::LLM => Ok(generator
            .generate_llm(
                segment_id,
                image_folder_location,
                segment_image.unwrap(),
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
    generation_strategy: &GenerationStrategy,
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
            generation_strategy,
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
    generation_strategy: &GenerationStrategy,
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
            generation_strategy,
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
    llm_prompt: Option<String>,
    llm_fallback_content: Option<String>,
    image_folder_location: &str,
    configuration: &Configuration,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    if llm_prompt.is_none() || segment_image.is_none() {
        return Ok(None);
    }

    let mut values = HashMap::new();
    let file_url = get_file_url(
        segment_image.as_ref().unwrap(),
        &format!("{}/{}.jpg", image_folder_location, segment.segment_id),
    )
    .await?;
    values.insert("segment_type".to_string(), segment.segment_type.to_string());
    values.insert("user_prompt".to_string(), llm_prompt.unwrap());
    values.insert("image_url".to_string(), file_url);

    let messages = create_messages_from_template("llm_segment", &values)?;
    let result = llm::try_extract_from_llm(
        messages,
        None,
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
    image_folder_location: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (html_strategy, markdown_strategy, llm_prompt) = match segment.segment_type.clone() {
        SegmentType::Table | SegmentType::Formula | SegmentType::Page => {
            let config: &LlmGenerationConfig = match segment.segment_type {
                SegmentType::Table => configuration.segment_processing.table.as_ref().unwrap(),
                SegmentType::Formula => configuration.segment_processing.formula.as_ref().unwrap(),
                SegmentType::Page => configuration.segment_processing.page.as_ref().unwrap(),
                _ => unreachable!(),
            };
            (&config.html, &config.markdown, &config.llm)
        }
        SegmentType::Picture => {
            let config: &PictureGenerationConfig =
                configuration.segment_processing.picture.as_ref().unwrap();
            (&config.html, &config.markdown, &config.llm)
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
            (&config.html, &config.markdown, &config.llm)
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

    let error_handling = match configuration.error_handling.clone() {
        Some(error_handling) => error_handling,
        None => ErrorHandlingStrategy::Fail,
    };

    // Create generators to use for auto-generation fallbacks
    let html_generator = HtmlGenerator {
        segment_type: segment.segment_type.clone(),
    };

    let markdown_generator = MarkdownGenerator {
        segment_type: segment.segment_type.clone(),
    };

    // Process HTML with error handling
    let html = match generate_html(
        segment,
        segment_image.clone(),
        html_strategy,
        fallback_html,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if error_handling == ErrorHandlingStrategy::Continue {
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
        markdown_strategy,
        fallback_markdown,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if error_handling == ErrorHandlingStrategy::Continue {
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
        llm_prompt.clone(),
        fallback_llm,
        image_folder_location,
        configuration,
    )
    .await
    {
        Ok(content) => content,
        Err(e) => {
            if error_handling == ErrorHandlingStrategy::Continue {
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
    let futures: Vec<_> = pipeline
        .chunks
        .iter_mut()
        .flat_map(|chunk| {
            chunk.segments.iter_mut().map(|segment| {
                process_segment(
                    segment,
                    &configuration,
                    segment_images.get(&segment.segment_id).map(|r| r.clone()),
                    &task.image_folder_location,
                )
            })
        })
        .collect();

    match futures::future::try_join_all(futures).await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error processing segments: {:?}", e);
            return Err(e.to_string().into());
        }
    }

    Ok(())
}

use crate::configs::llm_config::get_prompt;
use crate::models::chunkr::output::{Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::segment_processing::{
    AutoGenerationConfig, GenerationStrategy, LlmGenerationConfig, PictureGenerationConfig,
};
use crate::models::chunkr::task::{Configuration, Status};
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
    fn prompt_key(&self) -> &'static str;
    fn process_llm_result(&self, content: &str) -> String {
        content.to_string()
    }
    fn segment_type(&self) -> SegmentType;
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

    fn prompt_key(&self) -> &'static str {
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

    fn process_llm_result(&self, content: &str) -> String {
        match self.segment_type {
            SegmentType::Formula => format!("<span class=\"formula\">{}</span>", content),
            _ => content.to_string(),
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

    fn prompt_key(&self) -> &'static str {
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

    fn process_llm_result(&self, content: &str) -> String {
        match self.segment_type {
            SegmentType::Formula => format!("${}$", content),
            _ => content.to_string(),
        }
    }
}

async fn generate_content<T: ContentGenerator>(
    generator: &T,
    content: &str,
    override_content: String,
    segment_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if !override_content.is_empty() && generation_strategy == &GenerationStrategy::Auto {
        return Ok(override_content);
    }

    if segment_image.is_none() {
        return Ok(generator.generate_auto(content));
    }

    match generation_strategy {
        GenerationStrategy::LLM => {
            let prompt = get_prompt(generator.prompt_key(), &HashMap::new())?;
            let result = match (generator.prompt_key(), generator.segment_type()) {
                (_, SegmentType::Formula) => {
                    llm::latex_ocr(segment_image.as_ref().unwrap(), prompt).await?
                }
                (key, _) if key.starts_with("md_") => {
                    llm::markdown_ocr(segment_image.as_ref().unwrap(), prompt).await?
                }
                _ => llm::html_ocr(segment_image.as_ref().unwrap(), prompt).await?,
            };

            Ok(generator.process_llm_result(&result))
        }
        GenerationStrategy::Auto => Ok(generator.generate_auto(content)),
    }
}

async fn generate_html(
    segment_type: SegmentType,
    content: String,
    override_content: String,
    segment_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = HtmlGenerator { segment_type };
    Ok(html::clean_img_tags(
        &generate_content(
            &generator,
            &content,
            override_content,
            segment_image,
            generation_strategy,
        )
        .await?,
    ))
}

async fn generate_markdown(
    segment_type: SegmentType,
    content: String,
    override_content: String,
    segment_image: Option<Arc<NamedTempFile>>,
    generation_strategy: &GenerationStrategy,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let generator = MarkdownGenerator { segment_type };
    Ok(markdown::clean_img_tags(
        &generate_content(
            &generator,
            &content,
            override_content,
            segment_image,
            generation_strategy,
        )
        .await?,
    ))
}

async fn generate_llm(
    segment_type: SegmentType,
    segment_image: Option<Arc<NamedTempFile>>,
    llm_prompt: Option<String>,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    if llm_prompt.is_none() || segment_image.is_none() {
        return Ok(None);
    }

    let mut values = HashMap::new();
    values.insert("segment_type".to_string(), segment_type.to_string());
    values.insert("user_prompt".to_string(), llm_prompt.unwrap());
    let prompt = get_prompt("llm_segment", &values)?;
    let result = llm::llm_segment(segment_image.as_ref().unwrap(), prompt).await?;

    Ok(Some(result))
}

async fn process_segment(
    segment: &mut Segment,
    configuration: &Configuration,
    segment_image: Option<Arc<NamedTempFile>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (html_strategy, markdown_strategy, llm_prompt) = match segment.segment_type.clone() {
        SegmentType::Table | SegmentType::Formula => {
            let config: &LlmGenerationConfig = match segment.segment_type {
                SegmentType::Table => &configuration.segment_processing.table.as_ref().unwrap(),
                SegmentType::Formula => &configuration.segment_processing.formula.as_ref().unwrap(),
                _ => unreachable!(),
            };
            (&config.html, &config.markdown, &config.llm)
        }
        SegmentType::Picture => {
            let config: &PictureGenerationConfig =
                &configuration.segment_processing.picture.as_ref().unwrap();
            (&config.html, &config.markdown, &config.llm)
        }
        segment_type => {
            let config: &AutoGenerationConfig = match segment_type {
                SegmentType::Title => &configuration.segment_processing.title.as_ref().unwrap(),
                SegmentType::SectionHeader => &configuration
                    .segment_processing
                    .section_header
                    .as_ref()
                    .unwrap(),
                SegmentType::Text => &configuration.segment_processing.text.as_ref().unwrap(),
                SegmentType::ListItem => {
                    &configuration.segment_processing.list_item.as_ref().unwrap()
                }
                SegmentType::Caption => &configuration.segment_processing.caption.as_ref().unwrap(),
                SegmentType::Footnote => {
                    &configuration.segment_processing.footnote.as_ref().unwrap()
                }
                SegmentType::PageHeader => &configuration
                    .segment_processing
                    .page_header
                    .as_ref()
                    .unwrap(),
                SegmentType::PageFooter => &configuration
                    .segment_processing
                    .page_footer
                    .as_ref()
                    .unwrap(),
                SegmentType::Page => &configuration.segment_processing.page.as_ref().unwrap(),
                _ => unreachable!(),
            };
            (&config.html, &config.markdown, &config.llm)
        }
    };

    let (html, markdown, llm) = futures::try_join!(
        generate_html(
            segment.segment_type.clone(),
            segment.content.clone(),
            segment.html.clone(),
            segment_image.clone(),
            html_strategy
        ),
        generate_markdown(
            segment.segment_type.clone(),
            segment.content.clone(),
            segment.markdown.clone(),
            segment_image.clone(),
            markdown_strategy
        ),
        generate_llm(
            segment.segment_type.clone(),
            segment_image.clone(),
            llm_prompt.clone()
        )
    )?;

    segment.html = html;
    segment.markdown = markdown;
    segment.llm = llm;
    Ok(())
}

/// Process the segments and creates the html, llm and markdown fields
///
/// This function will generate the html, llm and markdown fields for all the segments in parallel.
/// Depending on the configuration, each segment will either be processed using heuristic or by a LLM.
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .get_task()?
        .update(
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

    pipeline.chunks.iter_mut().for_each(|chunk| {
        chunk.generate_embed_text();
    });

    Ok(())
}

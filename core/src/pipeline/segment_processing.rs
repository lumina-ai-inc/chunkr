use crate::configs::llm_config::get_prompt;
use crate::models::chunkr::output::{Segment, SegmentType};
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::segment_processing::{
    AutoGenerationConfig, GenerationStrategy, LlmGenerationConfig,
};
use crate::models::chunkr::task::{Configuration, Status};
use crate::utils::services::llm;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use tempfile::NamedTempFile;

lazy_static! {
    static ref NUMBERED_LIST_REGEX: Regex = Regex::new(r"^(\d+)\.\s+(.+)$").unwrap();
}

// TODO: Use static maps to make this easier to read and also allow markdown generation
async fn generate_html(
    segment_type: SegmentType,
    content: String,
    segment_image: Option<NamedTempFile>,
    generation_strategy: &GenerationStrategy,
) -> Result<String, Box<dyn std::error::Error>> {
    let html = match segment_type {
        SegmentType::Caption => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_caption", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<span class='caption'>{}</span>", content),
        },
        SegmentType::Footnote => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_footnote", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<span class='footnote'>{}</span>", content),
        },
        SegmentType::Formula => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("formula", &HashMap::new()).unwrap();
                match llm::latex_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(latex) => format!("<span class='formula'>{}</span>", latex),
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<span class='formula'>{}</span>", content),
        },
        SegmentType::ListItem => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_list_item", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => {
                if let Some(captures) = NUMBERED_LIST_REGEX.captures(content.trim()) {
                    let start_number = captures.get(1).unwrap().as_str().parse::<i32>().unwrap();
                    let item = captures.get(2).unwrap().as_str();
                    format!("<ol start='{}'><li>{}</li></ol>", start_number, item)
                } else {
                    let cleaned_content = content
                        .trim_start_matches(&['-', '*', '•', '●', ' '][..])
                        .trim();
                    format!("<ul><li>{}</li></ul>", cleaned_content)
                }
            }
        },
        SegmentType::Page => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_page", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<div class='page'>{}</div>", content),
        },
        SegmentType::PageFooter => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_page_footer", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<div class='page-footer'>{}</div>", content),
        },
        SegmentType::PageHeader => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_page_header", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<div class='page-header'>{}</div>", content),
        },
        SegmentType::Picture => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_picture", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => "<img src='' alt='{}' />".to_string(),
        },
        SegmentType::SectionHeader => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_section_header", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<h2>{}</h2>", content),
        },
        SegmentType::Table => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_table", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<table><tr><td>{}</td></tr></table>", content),
        },
        SegmentType::Text => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_text", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<p>{}</p>", content),
        },
        SegmentType::Title => match generation_strategy {
            GenerationStrategy::LLM => {
                let prompt = get_prompt("html_title", &HashMap::new()).unwrap();
                match llm::html_ocr(&segment_image.unwrap(), prompt).await {
                    Ok(html) => html,
                    Err(e) => {
                        return Err(e.to_string().into());
                    }
                }
            }
            GenerationStrategy::Auto => format!("<h1>{}</h1>", content),
        },
    };
    Ok(html)
}

async fn process_segment(
    segment: &mut Segment,
    configuration: &Configuration,
) -> Result<(), Box<dyn std::error::Error>> {
    match segment.segment_type {
        SegmentType::Table | SegmentType::Formula => {
            let config: &LlmGenerationConfig = match segment.segment_type {
                SegmentType::Table => &configuration.segment_processing.table,
                SegmentType::Formula => &configuration.segment_processing.formula,
                _ => unreachable!(),
            };
            match config.html {
                GenerationStrategy::LLM => {
                    segment.generate_html();
                }
                GenerationStrategy::Auto => {
                    segment.generate_html();
                }
            }
            match config.markdown {
                GenerationStrategy::LLM => {
                    segment.generate_markdown();
                }
                GenerationStrategy::Auto => {
                    segment.generate_markdown();
                }
            }
        }
        _ => {
            let config: &AutoGenerationConfig = match segment.segment_type {
                SegmentType::Title => &configuration.segment_processing.title,
                SegmentType::SectionHeader => &configuration.segment_processing.section_header,
                SegmentType::Text => &configuration.segment_processing.text,
                SegmentType::ListItem => &configuration.segment_processing.list_item,
                SegmentType::Picture => &configuration.segment_processing.picture,
                SegmentType::Caption => &configuration.segment_processing.caption,
                SegmentType::Footnote => &configuration.segment_processing.footnote,
                SegmentType::PageHeader => &configuration.segment_processing.page_header,
                SegmentType::PageFooter => &configuration.segment_processing.page_footer,
                SegmentType::Page => &configuration.segment_processing.page,
                _ => unreachable!(),
            };
            match config.html {
                GenerationStrategy::LLM => {
                    segment.generate_html();
                }
                GenerationStrategy::Auto => {
                    segment.generate_html();
                }
            }
            match config.markdown {
                GenerationStrategy::LLM => {
                    segment.generate_markdown();
                }
                GenerationStrategy::Auto => {
                    segment.generate_markdown();
                }
            }
        }
    };
    Ok(())
}

/// Process the segements and creates the html, llm and markdown fields
///
/// This function will generate the html, llm and markdown fields for all the segments in parallel.
/// Depending on the configruation, each segment will either be processed using hueristic or by a LLM.
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    pipeline
        .update_status(Status::Processing, Some("Processing segments".to_string()))
        .await?;

    let configuration = pipeline
        .task_payload
        .as_ref()
        .unwrap()
        .current_configuration
        .clone();

    let futures: Vec<_> = pipeline
        .chunks
        .as_mut()
        .unwrap()
        .iter_mut()
        .flat_map(|chunk| {
            chunk
                .segments
                .iter_mut()
                .map(|segment| process_segment(segment, &configuration))
        })
        .collect();

    futures::future::try_join_all(futures).await?;

    Ok(())
}

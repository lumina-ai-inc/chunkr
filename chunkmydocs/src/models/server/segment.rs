use crate::utils::configs::worker_config;
use crate::utils::services::structured_extraction::ExtractedJson;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// Bounding box for an item. It is used for both segments and OCR results.
pub struct BoundingBox {
    /// The left coordinate of the bounding box.
    pub left: f32,
    /// The top coordinate of the bounding box.
    pub top: f32,
    /// The width of the bounding box.
    pub width: f32,
    /// The height of the bounding box.
    pub height: f32,
}

impl BoundingBox {
    pub fn get_center(&self) -> (f32, f32) {
        (self.left + self.width / 2.0, self.top + self.height / 2.0)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// OCR results for a segment
pub struct OCRResult {
    pub bbox: BoundingBox,
    /// The recognized text of the OCR result.
    pub text: String,
    /// The confidence score of the recognized text.
    pub confidence: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Segment {
    /// Unique identifier for the segment.
    pub segment_id: String,
    pub bbox: BoundingBox,
    /// Page number of the segment.
    pub page_number: u32,
    /// Width of the page containing the segment.
    pub page_width: f32,
    /// Height of the page containing the segment.
    pub page_height: f32,
    /// Text content of the segment.
    pub content: String,
    pub segment_type: SegmentType,
    /// OCR results for the segment.
    pub ocr: Option<Vec<OCRResult>>,
    /// Presigned URL to the image of the segment.
    pub image: Option<String>,
    /// HTML representation of the segment.
    pub html: Option<String>,
    /// Markdown representation of the segment.
    pub markdown: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    /// Collection of document segments that form this chunk.
    /// When target_chunk_length > 0, contains the maximum number of segments
    /// that fit within that length (segments remain intact).
    /// Otherwise, contains exactly one segment.
    pub segments: Vec<Segment>,
    /// The total number of words in the chunk.
    pub chunk_length: i32,
}

// TODO: Move to models/server/task.rs
#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// The processed results of a document analysis task
pub struct OutputResponse {
    /// Collection of document chunks, where each chunk contains one or more segments
    pub chunks: Vec<Chunk>,
    pub extracted_json: Option<ExtractedJson>,
}

#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, EnumString, Display, ToSchema, ToSql, FromSql,
)]
/// The type a segment is classified as.
pub enum SegmentType {
    Title,
    #[serde(rename = "Section header")]
    SectionHeader,
    Text,
    #[serde(rename = "List item")]
    ListItem,
    Table,
    Picture,
    Caption,
    Formula,
    Footnote,
    #[serde(rename = "Page header")]
    PageHeader,
    #[serde(rename = "Page footer")]
    PageFooter,
    Page,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct PdlaSegment {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub page_number: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub text: String,
    #[serde(rename = "type")]
    pub segment_type: SegmentType,
}

impl PdlaSegment {
    pub fn to_segment(&self) -> Segment {
        Segment {
            segment_id: Uuid::new_v4().to_string(),
            bbox: BoundingBox {
                left: self.left,
                top: self.top,
                width: self.width,
                height: self.height,
            },
            page_number: self.page_number,
            page_width: self.page_width,
            page_height: self.page_height,
            content: self.text.clone(),
            segment_type: self.segment_type.clone(),
            ocr: None,
            image: None,
            html: None,
            markdown: None,
        }
    }
}

impl Segment {
    fn to_html(&self) -> String {
        match self.segment_type {
            SegmentType::Title => format!("<h1>{}</h1>", self.content),
            SegmentType::SectionHeader => format!("<h2>{}</h2>", self.content),
            SegmentType::Text => format!("<p>{}</p>", self.content),
            SegmentType::ListItem => {
                let parts = self.content.trim().split('.').collect::<Vec<&str>>();
                if parts[0].parse::<i32>().is_ok() {
                    let start_number = parts[0].parse::<i32>().unwrap();
                    let item = parts[1..].join(".").trim().to_string();
                    format!("<ol start='{}'><li>{}</li></ol>", start_number, item)
                } else {
                    let cleaned_content = self
                        .content
                        .trim_start_matches(&['-', '*', '•', '●', ' '][..])
                        .trim();
                    format!("<ul><li>{}</li></ul>", cleaned_content)
                }
            }
            SegmentType::Picture => "<img src='' alt='{}' />".to_string(),
            _ => format!(
                "<span class=\"{}\">{}</span>",
                self.segment_type
                    .to_string()
                    .to_lowercase()
                    .replace(" ", "-"),
                self.content
            ),
        }
    }

    fn to_markdown(&self) -> String {
        match self.segment_type {
            SegmentType::Title => format!("# {}\n\n", self.content),
            SegmentType::SectionHeader => format!("## {}\n\n", self.content),
            SegmentType::ListItem => {
                let parts = self.content.trim().split('.').collect::<Vec<&str>>();
                if parts[0].parse::<i32>().is_ok() {
                    let start_number = parts[0].parse::<i32>().unwrap();
                    let item: String = parts[1..].join(".").trim().to_string();
                    format!("{}. {}", start_number, item)
                } else {
                    let cleaned_content = self
                        .content
                        .trim_start_matches(&['-', '*', '•', '●', ' '][..])
                        .trim();
                    format!("- {}\n\n", cleaned_content)
                }
            }
            SegmentType::Picture => format!("![{}]()", self.segment_id),
            _ => format!("{}\n\n", self.content),
        }
    }

    fn update_content(&mut self) {
        if let Some(ocr) = &self.ocr {
            let config = match worker_config::Config::from_env() {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("Error getting extraction config: {}", e);
                    return;
                }
            };
            let avg_confidence = ocr
                .iter()
                .map(|ocr_result| ocr_result.confidence.unwrap_or(0.0))
                .sum::<f32>()
                / ocr.len() as f32;
            if avg_confidence >= config.ocr_confidence_threshold {
                self.content = ocr
                    .iter()
                    .map(|ocr_result| ocr_result.text.clone())
                    .collect::<Vec<String>>()
                    .join(" ");
            }
        }
    }

    pub fn finalize(&mut self) {
        self.update_content();
        self.html = Some(self.to_html());
        self.markdown = Some(self.to_markdown());
    }
}

use crate::models::chunkr::structured_extraction::ExtractedJson;
use lazy_static::lazy_static;
use postgres_types::{FromSql, ToSql};
use regex::Regex;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

lazy_static! {
    static ref NUMBERED_LIST_REGEX: Regex = Regex::new(r"^(\d+)\.\s+(.+)$").unwrap();
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// The processed results of a document analysis task
pub struct OutputResponse {
    /// Collection of document chunks, where each chunk contains one or more segments
    pub chunks: Vec<Chunk>,
    pub extracted_json: Option<ExtractedJson>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    pub chunk_id: String,
    /// The total number of words in the chunk.
    pub chunk_length: i32,
    /// Collection of document segments that form this chunk.
    /// When `target_chunk_length` > 0, contains the maximum number of segments
    /// that fit within that length (segments remain intact).
    /// Otherwise, contains exactly one segment.
    pub segments: Vec<Segment>,
}

impl Chunk {
    pub fn new(segments: Vec<Segment>) -> Self {
        let chunk_id = uuid::Uuid::new_v4().to_string();
        let chunk_length = segments
            .iter()
            .map(|s| s.content.split_whitespace().count())
            .sum::<usize>() as i32;
        Self {
            chunk_id,
            chunk_length,
            segments,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Segment {
    pub bbox: BoundingBox,
    /// Text content of the segment.
    pub content: String,
    /// Height of the page containing the segment.
    pub page_height: f32,
    /// HTML representation of the segment.
    pub html: Option<String>,
    /// Presigned URL to the image of the segment.
    pub image: Option<String>,
    /// Markdown representation of the segment.
    pub markdown: Option<String>,
    /// OCR results for the segment.
    pub ocr: Vec<OCRResult>,
    /// Page number of the segment.
    pub page_number: u32,
    /// Width of the page containing the segment.
    pub page_width: f32,
    /// Unique identifier for the segment.
    pub segment_id: String,
    pub segment_type: SegmentType,
}

impl Segment {
    pub fn new(
        bbox: BoundingBox,
        ocr_results: Vec<OCRResult>,
        page_height: f32,
        page_number: u32,
        page_width: f32,
        segment_type: SegmentType,
    ) -> Self {
        let segment_id = uuid::Uuid::new_v4().to_string();
        let content = ocr_results
            .iter()
            .map(|ocr_result| ocr_result.text.clone())
            .collect::<Vec<String>>()
            .join(" ");
        Self {
            segment_id,
            bbox,
            page_number,
            page_width,
            page_height,
            content,
            segment_type,
            ocr: ocr_results,
            image: None,
            html: None,
            markdown: None,
        }
    }

    pub fn new_from_page_ocr(
        bbox: BoundingBox,
        ocr_results: Vec<OCRResult>,
        page_height: f32,
        page_number: u32,
        page_width: f32,
        segment_type: SegmentType,
    ) -> Self {
        let segment_ocr: Vec<OCRResult> = ocr_results
            .into_iter()
            .filter(|ocr| ocr.bbox.intersects(&bbox))
            .map(|mut ocr| {
                ocr.bbox.left -= bbox.left;
                ocr.bbox.top -= bbox.top;
                ocr
            })
            .collect();

        Self::new(
            bbox,
            segment_ocr,
            page_height,
            page_number,
            page_width,
            segment_type,
        )
    }

    fn to_html(&self) -> String {
        match self.segment_type {
            SegmentType::Title => format!("<h1>{}</h1>", self.content),
            SegmentType::SectionHeader => format!("<h2>{}</h2>", self.content),
            SegmentType::Text => format!("<p>{}</p>", self.content),
            SegmentType::ListItem => {
                if let Some(captures) = NUMBERED_LIST_REGEX.captures(self.content.trim()) {
                    let start_number = captures.get(1).unwrap().as_str().parse::<i32>().unwrap();
                    let item = captures.get(2).unwrap().as_str();
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
                if let Some(captures) = NUMBERED_LIST_REGEX.captures(self.content.trim()) {
                    let start_number = captures.get(1).unwrap().as_str().parse::<i32>().unwrap();
                    let item = captures.get(2).unwrap().as_str();
                    format!("{}. {}\n\n", start_number, item)
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

    pub fn finalize(&mut self) {
        self.html = Some(self.to_html());
        self.markdown = Some(self.to_markdown());
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// Bounding box for an item. It is used for chunks, segments and OCR results.
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
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    pub fn get_center(&self) -> (f32, f32) {
        (self.left + self.width / 2.0, self.top + self.height / 2.0)
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        if self.left + self.width < other.left || other.left + other.width < self.left {
            return false;
        }

        if self.top + self.height < other.top || other.top + other.height < self.top {
            return false;
        }

        true
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

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    PartialEq,
    EnumString,
    Display,
    ToSchema,
    ToSql,
    FromSql,
    Eq,
    Hash,
)]
/// All the possible types for a segment.
/// Note: Different configurations will produce different types.
/// Please refer to the documentation for more information.
pub enum SegmentType {
    Title,
    #[serde(alias = "Section header")]
    SectionHeader,
    Text,
    #[serde(alias = "List item")]
    ListItem,
    Table,
    Picture,
    Caption,
    Formula,
    Footnote,
    #[serde(alias = "Page header")]
    PageHeader,
    #[serde(alias = "Page footer")]
    PageFooter,
    Page,
}

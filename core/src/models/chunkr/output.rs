use crate::models::chunkr::search::SimpleChunk;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::error::Error;
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn generate_content() -> String {
    String::new()
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// The processed results of a document analysis task
pub struct OutputResponse {
    /// Collection of document chunks, where each chunk contains one or more segments
    pub chunks: Vec<Chunk>,
    /// The name of the file.
    pub file_name: Option<String>,
    /// The number of pages in the file.
    pub page_count: Option<u32>,
    /// The presigned URL of the PDF file.
    pub pdf_url: Option<String>,
    #[deprecated]
    /// The extracted JSON from the document.
    pub extracted_json: Option<serde_json::Value>,
}

impl Default for OutputResponse {
    fn default() -> Self {
        Self {
            chunks: vec![],
            file_name: None,
            page_count: None,
            pdf_url: None,
            extracted_json: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    #[serde(default = "generate_uuid")]
    /// The unique identifier for the chunk.
    pub chunk_id: String,
    /// The total number of words in the chunk.
    pub chunk_length: u32,
    /// Collection of document segments that form this chunk.
    /// When `target_chunk_length` > 0, contains the maximum number of segments
    /// that fit within that length (segments remain intact).
    /// Otherwise, contains exactly one segment.
    pub segments: Vec<Segment>,
    /// Suggested text to be embed for search.
    pub embed: Option<String>,
}

impl Chunk {
    pub fn new(segments: Vec<Segment>) -> Self {
        let chunk_id = uuid::Uuid::new_v4().to_string();
        Self {
            chunk_id,
            chunk_length: 0,
            segments,
            embed: None,
        }
    }

    pub fn generate_embed_text(&mut self) {
        self.embed = Some(
            self.segments
                .iter()
                .map(|s| {
                    if !s.markdown.is_empty() {
                        s.markdown.clone()
                    } else if !s.html.is_empty() {
                        s.html.clone()
                    } else {
                        s.content.clone()
                    }
                })
                .collect::<Vec<String>>()
                .join(" "),
        );
        self.chunk_length = self.embed.as_ref().unwrap().split_whitespace().count() as u32;
    }

    /// Converts this Chunk into a SimpleChunk, containing just the ID and embed content
    pub fn to_simple(&self) -> Result<SimpleChunk, Box<dyn Error + Send + Sync>> {
        if self.embed.is_none() {
            return Err("Embed is not generated".into());
        }

        Ok(SimpleChunk {
            id: self.chunk_id.clone(),
            content: self.embed.clone().unwrap_or_default().to_string(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Segment {
    pub bbox: BoundingBox,
    /// Confidence score of the layout analysis model
    pub confidence: Option<f32>,
    #[serde(default = "generate_content")]
    /// Text content of the segment.
    pub content: String,
    #[serde(default = "generate_content")]
    /// HTML representation of the segment.
    pub html: String,
    /// Presigned URL to the image of the segment.
    pub image: Option<String>,
    /// LLM representation of the segment.
    pub llm: Option<String>,
    #[serde(default = "generate_content")]
    /// Markdown representation of the segment.
    pub markdown: String,
    /// OCR results for the segment.
    pub ocr: Option<Vec<OCRResult>>,
    /// Height of the page containing the segment.
    pub page_height: f32,
    /// Width of the page containing the segment.
    pub page_width: f32,
    /// Page number of the segment.
    pub page_number: u32,
    /// Unique identifier for the segment.
    pub segment_id: String,
    pub segment_type: SegmentType,
}

impl Segment {
    pub fn new(
        bbox: BoundingBox,
        confidence: Option<f32>,
        ocr_results: Vec<OCRResult>,
        page_height: f32,
        page_width: f32,
        page_number: u32,
        segment_type: SegmentType,
    ) -> Self {
        let segment_id = generate_uuid();
        let content = ocr_results
            .iter()
            .map(|ocr_result| ocr_result.text.clone())
            .collect::<Vec<String>>()
            .join(" ");
        Self {
            bbox,
            confidence,
            content,
            llm: None,
            page_height,
            page_number,
            page_width,
            segment_id,
            segment_type,
            ocr: Some(ocr_results),
            image: None,
            html: String::new(),
            markdown: String::new(),
        }
    }

    pub fn new_from_segment_ocr(
        bbox: BoundingBox,
        confidence: Option<f32>,
        segment_ocr: Vec<OCRResult>,
        page_height: f32,
        page_number: u32,
        page_width: f32,
        segment_type: SegmentType,
    ) -> Self {
        Self::new(
            bbox,
            confidence,
            segment_ocr,
            page_height,
            page_width,
            page_number,
            segment_type,
        )
    }

    pub fn scale(&mut self, scaling_factor: f32) {
        self.bbox.scale(scaling_factor);

        self.page_width *= scaling_factor;
        self.page_height *= scaling_factor;

        if let Some(ocr_results) = &mut self.ocr {
            for ocr_result in ocr_results {
                ocr_result.bbox.scale(scaling_factor);
            }
        }
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

    fn intersects(&self, other: &BoundingBox) -> bool {
        if self.left + self.width < other.left || other.left + other.width < self.left {
            return false;
        }

        if self.top + self.height < other.top || other.top + other.height < self.top {
            return false;
        }

        true
    }

    pub fn intersection_area(&self, other: &BoundingBox) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let x_left = self.left.max(other.left);
        let x_right = (self.left + self.width).min(other.left + other.width);
        let y_top = self.top.max(other.top);
        let y_bottom = (self.top + self.height).min(other.top + other.height);

        (x_right - x_left) * (y_bottom - y_top)
    }

    pub fn scale(&mut self, scaling_factor: f32) {
        self.left *= scaling_factor;
        self.top *= scaling_factor;
        self.width *= scaling_factor;
        self.height *= scaling_factor;
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
    Caption,
    Footnote,
    Formula,
    #[serde(alias = "List item")]
    ListItem,
    Page,
    #[serde(alias = "Page footer")]
    PageFooter,
    #[serde(alias = "Page header")]
    PageHeader,
    Picture,
    #[serde(alias = "Section header")]
    SectionHeader,
    Table,
    Text,
    Title,
}

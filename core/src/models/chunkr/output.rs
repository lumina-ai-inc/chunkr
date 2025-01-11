use crate::models::chunkr::structured_extraction::ExtractedJson;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// The processed results of a document analysis task
pub struct OutputResponse {
    /// Collection of document chunks, where each chunk contains one or more segments
    pub chunks: Vec<Chunk>,
    pub extracted_json: Option<ExtractedJson>,
}

impl Default for OutputResponse {
    fn default() -> Self {
        Self {
            chunks: vec![],
            extracted_json: None,
        }
    }
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
    // Confidence score of the segment
    pub confidence: Option<f32>,
    /// Text content of the segment.
    pub content: String,
    /// HTML representation of the segment.
    pub html: Option<String>,
    /// Presigned URL to the image of the segment.
    pub image: Option<String>,
    /// Markdown representation of the segment.
    pub markdown: Option<String>,
    /// OCR results for the segment.
    pub ocr: Vec<OCRResult>,
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
        let segment_id = uuid::Uuid::new_v4().to_string();
        let content = ocr_results
            .iter()
            .map(|ocr_result| ocr_result.text.clone())
            .collect::<Vec<String>>()
            .join(" ");
        Self {
            bbox,
            confidence,
            content,
            page_height,
            page_number,
            page_width,
            segment_id,
            segment_type,
            ocr: ocr_results,
            image: None,
            html: None,
            markdown: None,
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

use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct BoundingBox {
    pub top_left: Vec<f32>,
    pub top_right: Vec<f32>,
    pub bottom_right: Vec<f32>,
    pub bottom_left: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct OCRResult {
    pub bbox: BoundingBox,
    pub text: String,
    pub confidence: Option<f32>,
}


#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, EnumString, Display, ToSchema, ToSql, FromSql,
)]
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
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct BaseSegment {
    pub segment_id: String,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub text: String,
    #[serde(rename = "type")]
    pub segment_type: SegmentType,
    pub page_number: u32,
    pub page_width: f32,
    pub page_height: f32,
}


#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Segment {
    pub segment_id: String,
    pub bbox: BoundingBox,
    pub page_number: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub text: String,
    #[serde(rename = "type")]
    pub segment_type: SegmentType,
    pub ocr: Option<Vec<OCRResult>>,
    pub image: Option<String>,
    pub html: Option<String>,
    pub markdown: Option<String>,
}

impl Segment {
    pub fn new(
        bbox: BoundingBox,
        page_number: u32,
        page_width: f32,
        page_height: f32,
        text: String,
        segment_type: SegmentType,
        ocr: Option<Vec<OCRResult>>,
        image: Option<String>,
        html: Option<String>,
        markdown: Option<String>,
    ) -> Self {
        Self {
            bbox,
            page_number,
            page_width,
            page_height,
            text,
            segment_type,
            segment_id: Uuid::new_v4().to_string(),
            ocr,
            image,
            html,
            markdown,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    pub segments: Vec<Segment>,
    pub chunk_length: i32,
}

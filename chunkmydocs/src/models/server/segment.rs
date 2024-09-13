use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, EnumString, Display, ToSchema)]
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
pub struct Segment {
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
    pub segment_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    pub segments: Vec<Segment>,
    pub markdown: String,
}
#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct PngPage {
    pub bb_id: String,
    pub base64_png: String,
}

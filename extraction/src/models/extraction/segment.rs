use serde::{Serialize, Deserialize};
use strum_macros::{ Display, EnumString };

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, EnumString, Display)]
pub enum SegmentType {
    Caption,
    Footnote,
    Formula,
    ListItem,
    PageFooter,
    PageHeader,
    Picture,
    SectionHeader,
    Table,
    Text,
    Title,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
}
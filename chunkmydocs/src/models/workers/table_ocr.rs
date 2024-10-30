use crate::models::server::segment::BoundingBox;
use serde::{ Deserialize, Serialize };

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Cell {
    pub column: BoundingBox,
    pub cell: BoundingBox,
    pub content: Option<String>,
    pub confidence: Option<f32>,
    pub col_span: i32,
    pub row_span: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TableStructure {
    pub row: BoundingBox,
    pub cells: Vec<Cell>,
    pub cell_count: i32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TableStructureResponse {
    pub result: Vec<TableStructure>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Table {
    pub bbox: Vec<f32>,
    pub html: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PaddleTableRecognitionResult {
    pub tables: Vec<Table>,
    #[serde(rename = "layoutImage")]
    pub layout_image: String,
    #[serde(rename = "ocrImage")]
    pub ocr_image: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PaddleTableRecognitionResponse {
    #[serde(rename = "logId")]
    pub log_id: String,
    #[serde(rename = "errorCode")]
    pub error_code: i32,
    #[serde(rename = "errorMsg")]
    pub error_msg: String,
    pub result: PaddleTableRecognitionResult,
}
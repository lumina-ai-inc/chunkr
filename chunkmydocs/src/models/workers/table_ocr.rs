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
pub struct PaddleTableRecognitionResponse {
    pub bbox: Vec<f32>,
    pub html: String,
}

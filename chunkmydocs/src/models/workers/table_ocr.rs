use crate::models::server::segment::BoundingBox;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Cell {
    pub column: BoundingBox,
    pub cell: BoundingBox,
    pub content: Option<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableStructure {
    pub row: BoundingBox,
    pub cells: Vec<Cell>,
    pub cell_count: i32,
    pub content: Option<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableStructureResponse {
    pub result: Vec<TableStructure>,
}

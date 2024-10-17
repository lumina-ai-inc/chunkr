use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

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

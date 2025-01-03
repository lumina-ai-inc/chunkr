use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct OcrWord {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub text: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Instance {
    pub boxes: Vec<BoundingBox>,
    pub scores: Vec<f32>,
    pub classes: Vec<i32>,
    pub image_size: (i32, i32),
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct ObjectDetectionResponse {
    pub instances: Instance,
}

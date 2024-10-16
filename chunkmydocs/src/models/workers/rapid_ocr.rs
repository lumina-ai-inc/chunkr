use crate::models::server::segment::{OCRResult, BoundingBox};
use serde::{ Deserialize, Serialize };

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RapidOcrPayload {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: [f32; 8],
}

impl From<RapidOcrPayload> for OCRResult {
    fn from(payload: RapidOcrPayload) -> Self {
        let [x1, y1, x2, y2, x3, y3, x4, y4] = payload.bounding_box;
        let left = x1.min(x4);
        let top = y1.min(y2);
        let right = x2.max(x3);
        let bottom = y3.max(y4);

        OCRResult {
            bbox: BoundingBox {
                left,
                top,
                width: right - left,
                height: bottom - top,
            },
            text: payload.text,
            confidence: Some(payload.confidence),
        }
    }
}
use crate::models::server::segment::{OCRResult, BoundingBox};
use serde::{ Deserialize, Serialize };

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RapidOcrPayload {
    pub result: Vec<PPOCRPayload>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PPOCRPayload {
    pub bounding_box: Vec<Vec<f32>>,
    pub text: String,
    pub confidence: f32,
}

impl From<PPOCRPayload> for OCRResult {
    fn from(payload: PPOCRPayload) -> Self {
        let bbox = &payload.bounding_box;
        let left = bbox[0][0].min(bbox[3][0]);
        let top = bbox[0][1].min(bbox[1][1]);
        let right = bbox[1][0].max(bbox[2][0]);
        let bottom = bbox[2][1].max(bbox[3][1]);

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
use crate::models::server::segment::{BoundingBox, OCRResult};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneralOcrResult {
    pub texts: Vec<Text>,
    pub image: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Text {
    pub poly: Vec<Vec<i32>>,
    pub text: String,
    pub score: f32,
}

impl From<Text> for OCRResult {
    fn from(payload: Text) -> Self {
        let bbox = &payload.poly;
        let left = bbox[0][0].min(bbox[3][0]) as f32;
        let top = bbox[0][1].min(bbox[1][1]) as f32;
        let right = bbox[1][0].max(bbox[2][0]) as f32;
        let bottom = bbox[2][1].max(bbox[3][1]) as f32;

        OCRResult {
            bbox: BoundingBox {
                left,
                top,
                width: right - left,
                height: bottom - top,
            },
            text: payload.text,
            confidence: Some(payload.score),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PaddleOCRResponse {
    #[serde(rename = "logId")]
    pub log_id: String,
    #[serde(rename = "errorCode")]
    pub error_code: i32,
    #[serde(rename = "errorMsg")]
    pub error_msg: String,
    pub result: GeneralOcrResult,
}

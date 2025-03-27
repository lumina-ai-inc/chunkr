use crate::models::output::{BoundingBox, OCRResult};
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

#[derive(Debug, Serialize, Deserialize)]
pub struct DoctrResponse {
    pub page_content: PageContent,
    pub processing_time: f64,
}

impl From<DoctrResponse> for Vec<OCRResult> {
    fn from(payload: DoctrResponse) -> Self {
        let mut results = Vec::new();

        for block in payload.page_content.blocks {
            for line in block.lines {
                for word in line.words {
                    let geometry = &word.geometry;
                    let left = geometry[0][0] as f32 * payload.page_content.dimensions[1] as f32;
                    let top = geometry[0][1] as f32 * payload.page_content.dimensions[0] as f32;
                    let right = geometry[1][0] as f32 * payload.page_content.dimensions[1] as f32;
                    let bottom = geometry[1][1] as f32 * payload.page_content.dimensions[0] as f32;

                    results.push(OCRResult {
                        bbox: BoundingBox {
                            left,
                            top,
                            width: right - left,
                            height: bottom - top,
                        },
                        text: word.value,
                        confidence: Some(word.confidence as f32),
                    });
                }
            }
        }

        results
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageContent {
    pub page_idx: i32,
    pub dimensions: Vec<i32>,
    pub orientation: Detection<Option<f64>>,
    pub language: Detection<Option<String>>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Detection<T> {
    pub value: T,
    pub confidence: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Block {
    pub geometry: Vec<Vec<f64>>,
    pub objectness_score: f64,
    pub lines: Vec<Line>,
    pub artefacts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Line {
    pub geometry: Vec<Vec<f64>>,
    pub objectness_score: f64,
    pub words: Vec<Word>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Word {
    pub value: String,
    pub confidence: f64,
    pub geometry: Vec<Vec<f64>>,
    pub objectness_score: f64,
    pub crop_orientation: Detection<i32>,
}

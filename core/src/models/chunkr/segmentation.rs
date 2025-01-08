use crate::models::chunkr::output::{BoundingBox, OCRResult, Segment, SegmentType};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Instance {
    pub boxes: Vec<BoundingBox>,
    pub scores: Vec<f32>,
    pub classes: Vec<i32>,
    pub image_size: (i32, i32),
}

impl Instance {
    fn get_segment_type(&self, index: usize) -> Option<SegmentType> {
        self.classes.get(index).map(|&class_idx| match class_idx {
            0 => SegmentType::Caption,
            1 => SegmentType::Footnote,
            2 => SegmentType::Formula,
            3 => SegmentType::ListItem,
            4 => SegmentType::PageFooter,
            5 => SegmentType::PageHeader,
            6 => SegmentType::Picture,
            7 => SegmentType::SectionHeader,
            8 => SegmentType::Table,
            9 => SegmentType::Text,
            10 => SegmentType::Title,
            _ => SegmentType::Text,
        })
    }

    pub fn to_segments(&self, page_number: u32, ocr_results: Vec<OCRResult>) -> Vec<Segment> {
        let (page_width, page_height) = (self.image_size.0 as f32, self.image_size.1 as f32);
        self.boxes
            .iter()
            .enumerate()
            .filter_map(|(idx, bbox)| {
                let confidence = self.scores.get(idx).copied().unwrap_or(0.0);
                self.get_segment_type(idx).map(|segment_type| {
                    Segment::new_from_page_ocr(
                        bbox.clone(),
                        Some(confidence),
                        ocr_results.clone(),
                        page_height,
                        page_number,
                        page_width,
                        segment_type,
                    )
                })
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct ObjectDetectionResponse {
    pub instances: Instance,
}

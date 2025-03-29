use crate::configs::worker_config;
use crate::models::output::{BoundingBox, OCRResult, Segment, SegmentType};
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
        let worker_config = worker_config::Config::from_env().unwrap();
        let (page_height, page_width) = (self.image_size.0 as f32, self.image_size.1 as f32);

        if self.boxes.is_empty() {
            println!(
                "No segments detected for page {}. Adding full-page segment with dimensions {:?}",
                page_number,
                (page_width, page_height)
            );

            return vec![Segment::new(
                BoundingBox::new(0.0, 0.0, page_width, page_height),
                Some(1.0),
                ocr_results,
                page_height,
                page_width,
                page_number,
                SegmentType::Page,
            )];
        }

        let padded_boxes: Vec<_> = self
            .boxes
            .iter()
            .map(|bbox| {
                let mut bbox = bbox.clone();
                bbox.top -= worker_config.segmentation_padding;
                bbox.left -= worker_config.segmentation_padding;
                bbox.width += worker_config.segmentation_padding * 2.0;
                bbox.height += worker_config.segmentation_padding * 2.0;
                bbox
            })
            .collect();

        let mut ocr_assignments: Vec<(usize, OCRResult)> = Vec::new();

        for ocr in ocr_results {
            let mut best_area = 0.0;
            let mut best_idx = None;

            for (idx, bbox) in padded_boxes.iter().enumerate() {
                let area = bbox.intersection_area(&ocr.bbox);
                if area > best_area {
                    best_area = area;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                ocr_assignments.push((idx, ocr));
            }
        }

        self.boxes
            .iter()
            .enumerate()
            .filter_map(|(idx, _)| {
                let confidence = self.scores.get(idx).copied().unwrap_or(0.0);
                let bbox = &padded_boxes[idx];

                let segment_ocr: Vec<OCRResult> = ocr_assignments
                    .iter()
                    .filter(|(assigned_idx, _)| *assigned_idx == idx)
                    .map(|(_, ocr)| {
                        let mut ocr = ocr.clone();
                        ocr.bbox.left -= bbox.left;
                        ocr.bbox.top -= bbox.top;
                        ocr
                    })
                    .collect();

                self.get_segment_type(idx).map(|segment_type| {
                    Segment::new(
                        bbox.clone(),
                        Some(confidence),
                        segment_ocr,
                        page_height,
                        page_width,
                        page_number,
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

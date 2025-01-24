pub mod auth;
pub mod chunk_processing;
pub mod cropping;
pub mod general_ocr;
pub mod open_ai;
pub mod output;
pub mod pipeline;
pub mod search;
pub mod segment_processing;
pub mod segmentation;
// pub mod structured_extraction;
pub mod task;
pub mod tasks;
pub mod upload;
pub mod user;

#[cfg(feature = "azure")]
pub mod azure;

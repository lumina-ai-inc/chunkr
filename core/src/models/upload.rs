use crate::configs::job_config;
use crate::models::chunk_processing::ChunkProcessing;
use crate::models::segment_processing::SegmentProcessing;
use crate::models::task::Configuration;
#[cfg(feature = "azure")]
use crate::models::task::PipelineType;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::{IntoParams, ToSchema};

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString, Default
)]
/// Controls the Optical Character Recognition (OCR) strategy.
/// - `All`: Processes all pages with OCR. (Latency penalty: ~0.5 seconds per page)
/// - `Auto`: Selectively applies OCR only to pages with missing or low-quality text. When text layer is present the bounding boxes from the text layer are used.
pub enum OcrStrategy {
    #[default]
    All,
    #[serde(alias = "Off")]
    Auto,
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Display,
    EnumString,
    Eq,
    PartialEq,
    ToSql,
    FromSql,
    ToSchema,
    Default,
)]
/// Controls the segmentation strategy:
/// - `LayoutAnalysis`: Analyzes pages for layout elements (e.g., `Table`, `Picture`, `Formula`, etc.) using bounding boxes. Provides fine-grained segmentation and better chunking. (Latency penalty: ~TBD seconds per page).
/// - `Page`: Treats each page as a single segment. Faster processing, but without layout element detection and only simple chunking.
pub enum SegmentationStrategy {
    #[default]
    LayoutAnalysis,
    Page,
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Display,
    EnumString,
    Eq,
    PartialEq,
    ToSql,
    FromSql,
    ToSchema,
    Default
)]
/// Controls how errors are handled during processing:
/// - `Fail`: Stops processing and fails the task when any error occurs
/// - `Continue`: Attempts to continue processing despite non-critical errors (eg. LLM refusals etc.)
pub enum ErrorHandlingStrategy {
    #[default]
    Fail,
    Continue,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, IntoParams)]
pub struct CreateForm {
    pub chunk_processing: Option<ChunkProcessing>,
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<i32>,
    /// The file to be uploaded. Can be a URL or a base64 encoded file.
    pub file: String,
    /// The name of the file to be uploaded. If not set a name will be generated.
    pub file_name: Option<String>,
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    #[schema(default = false)]
    pub high_resolution: Option<bool>,
    #[schema(default = "All")]
    pub ocr_strategy: Option<OcrStrategy>,
    #[cfg(feature = "azure")]
    #[schema(default = "Azure")]
    /// Choose the provider whose models will be used for segmentation and OCR.
    /// The output will be unified to the Chunkr `output` format.
    pub pipeline: Option<PipelineType>,
    pub segment_processing: Option<SegmentProcessing>,
    #[schema(default = "LayoutAnalysis")]
    pub segmentation_strategy: Option<SegmentationStrategy>,
    #[schema(default = "Fail")]
    /// Controls whether processing should stop on errors or attempt to continue
    pub error_handling: Option<ErrorHandlingStrategy>,
}

impl CreateForm {
    fn get_chunk_processing(&self) -> ChunkProcessing {
        self.chunk_processing
            .clone()
            .unwrap_or(ChunkProcessing::default())
    }

    fn get_expires_in(&self) -> Option<i32> {
        let job_config = job_config::Config::from_env().unwrap();
        self.expires_in.or(job_config.expiration_time)
    }

    fn get_high_resolution(&self) -> bool {
        self.high_resolution.unwrap_or(false)
    }

    fn get_ocr_strategy(&self) -> OcrStrategy {
        self.ocr_strategy.clone().unwrap_or_default()
    }

    fn get_segment_processing(&self) -> SegmentProcessing {
        let user_config = self.segment_processing.clone().unwrap_or_default();
        SegmentProcessing {
            title: user_config
                .title
                .or_else(|| SegmentProcessing::default().title),
            section_header: user_config
                .section_header
                .or_else(|| SegmentProcessing::default().section_header),
            text: user_config
                .text
                .or_else(|| SegmentProcessing::default().text),
            list_item: user_config
                .list_item
                .or_else(|| SegmentProcessing::default().list_item),
            table: user_config
                .table
                .or_else(|| SegmentProcessing::default().table),
            picture: user_config
                .picture
                .or_else(|| SegmentProcessing::default().picture),
            caption: user_config
                .caption
                .or_else(|| SegmentProcessing::default().caption),
            formula: user_config
                .formula
                .or_else(|| SegmentProcessing::default().formula),
            footnote: user_config
                .footnote
                .or_else(|| SegmentProcessing::default().footnote),
            page_header: user_config
                .page_header
                .or_else(|| SegmentProcessing::default().page_header),
            page_footer: user_config
                .page_footer
                .or_else(|| SegmentProcessing::default().page_footer),
            page: user_config
                .page
                .or_else(|| SegmentProcessing::default().page),
        }
    }

    fn get_segmentation_strategy(&self) -> SegmentationStrategy {
        self.segmentation_strategy.clone().unwrap_or_default()
    }

    #[cfg(feature = "azure")]
    fn get_pipeline(&self) -> Option<PipelineType> {
        Some(self.pipeline.clone().unwrap_or_default())
    }

    fn get_error_handling(&self) -> Option<ErrorHandlingStrategy> {
        Some(self.error_handling.clone().unwrap_or_default())
    }

    pub fn to_configuration(&self) -> Configuration {
        Configuration {
            chunk_processing: self.get_chunk_processing(),
            expires_in: self.get_expires_in(),
            high_resolution: self.get_high_resolution(),
            input_file_url: None,
            json_schema: None,
            model: None,
            ocr_strategy: self.get_ocr_strategy(),
            #[cfg(feature = "azure")]
            pipeline: self.get_pipeline(),
            segment_processing: self.get_segment_processing(),
            segmentation_strategy: self.get_segmentation_strategy(),
            target_chunk_length: None,
            error_handling: self.get_error_handling(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, IntoParams)]
pub struct UpdateForm {
    pub chunk_processing: Option<ChunkProcessing>,
    /// The number of seconds until task is deleted.
    /// Expired tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<i32>,
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    pub high_resolution: Option<bool>,
    pub ocr_strategy: Option<OcrStrategy>,
    #[cfg(feature = "azure")]
    /// Choose the provider whose models will be used for segmentation and OCR.
    /// The output will be unified to the Chunkr `output` format.
    pub pipeline: Option<PipelineType>,
    pub segment_processing: Option<SegmentProcessing>,
    pub segmentation_strategy: Option<SegmentationStrategy>,
    pub error_handling: Option<ErrorHandlingStrategy>,
}

impl UpdateForm {
    fn get_segment_processing(&self, current_config: &Configuration) -> SegmentProcessing {
        let user_config = self.segment_processing.clone().unwrap_or_default();

        SegmentProcessing {
            title: user_config
                .title
                .or(current_config.segment_processing.title.clone()),
            section_header: user_config
                .section_header
                .or(current_config.segment_processing.section_header.clone()),
            text: user_config
                .text
                .or(current_config.segment_processing.text.clone()),
            list_item: user_config
                .list_item
                .or(current_config.segment_processing.list_item.clone()),
            table: user_config
                .table
                .or(current_config.segment_processing.table.clone()),
            picture: user_config
                .picture
                .or(current_config.segment_processing.picture.clone()),
            caption: user_config
                .caption
                .or(current_config.segment_processing.caption.clone()),
            formula: user_config
                .formula
                .or(current_config.segment_processing.formula.clone()),
            footnote: user_config
                .footnote
                .or(current_config.segment_processing.footnote.clone()),
            page_header: user_config
                .page_header
                .or(current_config.segment_processing.page_header.clone()),
            page_footer: user_config
                .page_footer
                .or(current_config.segment_processing.page_footer.clone()),
            page: user_config
                .page
                .or(current_config.segment_processing.page.clone()),
        }
    }

    pub fn to_configuration(&self, current_config: &Configuration) -> Configuration {
        Configuration {
            chunk_processing: self
                .chunk_processing
                .clone()
                .unwrap_or_else(|| current_config.chunk_processing.clone()),
            expires_in: self.expires_in.or(current_config.expires_in),
            high_resolution: self
                .high_resolution
                .unwrap_or(current_config.high_resolution),
            input_file_url: None,
            json_schema: None,
            model: None,
            ocr_strategy: self
                .ocr_strategy
                .clone()
                .unwrap_or(current_config.ocr_strategy.clone()),
            #[cfg(feature = "azure")]
            pipeline: current_config.pipeline.clone(),
            segment_processing: self.get_segment_processing(current_config),
            segmentation_strategy: self
                .segmentation_strategy
                .clone()
                .unwrap_or(current_config.segmentation_strategy.clone()),
            target_chunk_length: None,
            error_handling: self
                .error_handling
                .clone()
                .or_else(|| current_config.error_handling.clone()),
        }
    }
}

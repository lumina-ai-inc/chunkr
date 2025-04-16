use crate::configs::job_config;
use crate::models::chunk_processing::ChunkProcessing;
use crate::models::llm::LlmProcessing;
use crate::models::segment_processing::SegmentProcessing;
use crate::models::task::Configuration;
#[cfg(feature = "azure")]
use crate::models::task::PipelineType;
use crate::models::upload::{ErrorHandlingStrategy, OcrStrategy, SegmentationStrategy};
use actix_multipart::form::json::Json as MPJson;
use actix_multipart::form::{tempfile::TempFile, MultipartForm};
use utoipa::{IntoParams, ToSchema};

#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct CreateFormMultipart {
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<ChunkProcessing>, format = "binary")]
    pub chunk_processing: Option<MPJson<ChunkProcessing>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<i32>, format = "binary")]
    /// The number of seconds until task is deleted.
    /// Expired tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<MPJson<i32>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    /// The file to be uploaded.
    pub file: TempFile,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<bool>, default = false, format = "binary")]
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    pub high_resolution: Option<MPJson<bool>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<OcrStrategy>, default = "All", format = "binary")]
    pub ocr_strategy: Option<MPJson<OcrStrategy>>,
    #[cfg(feature = "azure")]
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<PipelineType>, format = "binary")]
    /// The PipelineType to use for processing.
    /// If pipeline is set to Azure then Azure layout analysis will be used for segmentation and OCR.
    /// The output will be unified to the Chunkr `output` format.
    pub pipeline: Option<MPJson<PipelineType>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<SegmentProcessing>, format = "binary")]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = Option<SegmentationStrategy>, default = "LayoutAnalysis", format = "binary")]
    pub segmentation_strategy: Option<MPJson<SegmentationStrategy>>,
}

impl CreateFormMultipart {
    fn get_chunk_processing(&self) -> ChunkProcessing {
        self.chunk_processing
            .as_ref()
            .map(|mp_json| mp_json.0.clone())
            .unwrap_or_default()
    }

    fn get_expires_in(&self) -> Option<i32> {
        let job_config = job_config::Config::from_env().unwrap();
        self.expires_in
            .as_ref()
            .map(|e| e.0)
            .or(job_config.expiration_time)
    }

    fn get_high_resolution(&self) -> bool {
        self.high_resolution.as_ref().map(|e| e.0).unwrap_or(false)
    }

    fn get_ocr_strategy(&self) -> OcrStrategy {
        self.ocr_strategy
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_default()
    }

    fn get_segment_processing(&self) -> SegmentProcessing {
        let user_config = self
            .segment_processing
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_default();

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
        self.segmentation_strategy
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_default()
    }

    #[cfg(feature = "azure")]
    fn get_pipeline(&self) -> Option<PipelineType> {
        self.pipeline.as_ref().map(|e| e.0.clone())
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
            error_handling: ErrorHandlingStrategy::default(),
            llm_processing: LlmProcessing::default(),
        }
    }
}

#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct UpdateFormMultipart {
    #[param(style = Form, value_type = Option<ChunkProcessing>, format = "binary")]
    #[schema(value_type = Option<ChunkProcessing>, format = "binary")]
    pub chunk_processing: Option<MPJson<ChunkProcessing>>,
    #[param(style = Form, value_type = Option<i32>, format = "binary")]
    #[schema(value_type = Option<i32>, format = "binary")]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<MPJson<i32>>,
    #[param(style = Form, value_type = Option<bool>, format = "binary")]
    #[schema(value_type = Option<bool>, format = "binary")]
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    pub high_resolution: Option<MPJson<bool>>,
    #[param(style = Form, value_type = Option<OcrStrategy>, format = "binary")]
    #[schema(value_type = Option<OcrStrategy>, format = "binary")]
    pub ocr_strategy: Option<MPJson<OcrStrategy>>,
    #[cfg(feature = "azure")]
    #[param(style = Form, value_type = Option<PipelineType>, format = "binary")]
    #[schema(value_type = Option<PipelineType>, format = "binary")]
    /// The pipeline to use for processing.
    /// If pipeline is set to Azure then Azure layout analysis will be used for segmentation and OCR.
    /// The output will be unified to the Chunkr output.
    pub pipeline: Option<MPJson<PipelineType>>,
    #[param(style = Form, value_type = Option<SegmentProcessing>, format = "binary")]
    #[schema(value_type = Option<SegmentProcessing>, format = "binary")]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>, format = "binary")]
    #[schema(value_type = Option<SegmentationStrategy>, format = "binary")]
    pub segmentation_strategy: Option<MPJson<SegmentationStrategy>>,
}

impl UpdateFormMultipart {
    fn get_segment_processing(&self, current_config: &Configuration) -> SegmentProcessing {
        let user_config = self
            .segment_processing
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_default();

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
                .as_ref()
                .map(|e| e.0.clone())
                .unwrap_or_else(|| current_config.chunk_processing.clone()),
            expires_in: self
                .expires_in
                .as_ref()
                .map(|e| e.0)
                .or(current_config.expires_in),
            high_resolution: self
                .high_resolution
                .as_ref()
                .map(|e| e.0)
                .unwrap_or(current_config.high_resolution),
            input_file_url: None,
            json_schema: None,
            model: None,
            ocr_strategy: self
                .ocr_strategy
                .as_ref()
                .map(|e| e.0.clone())
                .unwrap_or(current_config.ocr_strategy.clone()),
            #[cfg(feature = "azure")]
            pipeline: self.pipeline.as_ref().map(|e| e.0.clone()),
            segment_processing: self.get_segment_processing(current_config),
            segmentation_strategy: self
                .segmentation_strategy
                .as_ref()
                .map(|e| e.0.clone())
                .unwrap_or(current_config.segmentation_strategy.clone()),
            target_chunk_length: None,
            error_handling: ErrorHandlingStrategy::default(),
            llm_processing: LlmProcessing::default(),
        }
    }
}

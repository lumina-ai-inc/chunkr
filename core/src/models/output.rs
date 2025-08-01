use crate::models::{
    chunk_processing::TokenizerType, search::SimpleChunk, segment_processing::EmbedSource,
    task::Configuration,
};
use lru::LruCache;
use once_cell::sync::Lazy;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use strum_macros::{Display, EnumString};
use tiktoken_rs::cl100k_base;
use tokenizers::tokenizer::Tokenizer;
use utoipa::ToSchema;

static WORD_COUNT_CACHE: Lazy<Arc<Mutex<LruCache<String, u32>>>> = Lazy::new(|| {
    let cache_size = NonZeroUsize::new(10000).unwrap();
    Arc::new(Mutex::new(LruCache::new(cache_size)))
});

fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn generate_string() -> String {
    String::new()
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, Default)]
/// The processed results of a document analysis task
pub struct OutputResponse {
    /// Collection of document chunks, where each chunk contains one or more segments
    pub chunks: Vec<Chunk>,
    /// The name of the file.
    pub file_name: Option<String>,
    /// The number of pages in the file.
    pub page_count: Option<u32>,
    /// The presigned URL of the PDF file.
    pub pdf_url: Option<String>,
    #[deprecated]
    /// The extracted JSON from the document.
    pub extracted_json: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Chunk {
    #[serde(default = "generate_uuid")]
    /// The unique identifier for the chunk.
    pub chunk_id: String,
    /// The total number of tokens in the chunk. Calculated by the `tokenizer`.
    pub chunk_length: u32,
    /// Collection of document segments that form this chunk.
    /// When `target_chunk_length` > 0, contains the maximum number of segments
    /// that fit within that length (segments remain intact).
    /// Otherwise, contains exactly one segment.
    pub segments: Vec<Segment>,
    /// Suggested text to be embedded for the chunk. This text is generated by combining the embed content
    /// from each segment according to the configured embed sources (HTML, Markdown, LLM, or Content).
    /// Can be configured using `embed_sources` in the `SegmentProcessing` configuration.
    pub embed: Option<String>,
}

impl Chunk {
    pub fn new(segments: Vec<Segment>) -> Self {
        let chunk_id = uuid::Uuid::new_v4().to_string();
        Self {
            chunk_id,
            chunk_length: 0,
            segments,
            embed: None,
        }
    }

    pub fn generate_embed_text(&mut self, configuration: &Configuration) {
        self.embed = Some(
            self.segments
                .iter()
                .map(|s| s.get_embed_content(configuration))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        self.chunk_length = self
            .segments
            .iter()
            .map(|s| s.count_embed_words(configuration))
            .filter_map(Result::ok)
            .sum();
    }

    /// Converts this Chunk into a SimpleChunk, containing just the ID and embed content
    pub fn to_simple(&self) -> std::result::Result<SimpleChunk, Box<dyn Error + Send + Sync>> {
        if self.embed.is_none() {
            return Err("Embed is not generated".into());
        }

        Ok(SimpleChunk {
            id: self.chunk_id.clone(),
            content: self.embed.clone().unwrap_or_default().to_string(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Segment {
    pub bbox: BoundingBox,
    /// Confidence score of the layout analysis model
    pub confidence: Option<f32>,
    /// Content of the segment, will be either HTML or Markdown, depending on format chosen.
    #[serde(default = "generate_string")]
    pub content: String,
    /// HTML representation of the segment.
    #[serde(default = "generate_string")]
    pub html: String,
    /// Presigned URL to the image of the segment.
    pub image: Option<String>,
    /// LLM representation of the segment.
    pub llm: Option<String>,
    #[serde(default = "generate_string")]
    /// Markdown representation of the segment.
    pub markdown: String,
    /// OCR results for the segment.
    pub ocr: Option<Vec<OCRResult>>,
    /// Height of the page containing the segment.
    pub page_height: f32,
    /// Width of the page containing the segment.
    pub page_width: f32,
    /// Page number of the segment.
    pub page_number: u32,
    /// Unique identifier for the segment.
    pub segment_id: String,
    pub segment_type: SegmentType,
    /// Text content of the segment. Calculated by the OCR results.
    #[serde(default = "generate_string")]
    pub text: String,
}

impl Segment {
    pub fn new(
        bbox: BoundingBox,
        confidence: Option<f32>,
        ocr_results: Vec<OCRResult>,
        page_height: f32,
        page_width: f32,
        page_number: u32,
        segment_type: SegmentType,
    ) -> Self {
        let segment_id = generate_uuid();
        let text = ocr_results
            .iter()
            .map(|ocr_result| ocr_result.text.clone())
            .collect::<Vec<String>>()
            .join(" ");
        Self {
            bbox,
            confidence,
            content: String::new(),
            llm: None,
            page_height,
            page_number,
            page_width,
            segment_id,
            segment_type,
            ocr: Some(ocr_results),
            image: None,
            html: String::new(),
            markdown: String::new(),
            text,
        }
    }

    pub fn scale(&mut self, scaling_factor: f32) {
        self.bbox.scale(scaling_factor);

        self.page_width *= scaling_factor;
        self.page_height *= scaling_factor;

        if let Some(ocr_results) = &mut self.ocr {
            for ocr_result in ocr_results {
                ocr_result.bbox.scale(scaling_factor);
            }
        }
    }

    fn get_embed_content(&self, configuration: &Configuration) -> String {
        let embed_sources = match self.segment_type {
            SegmentType::Title => configuration
                .segment_processing
                .title
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::SectionHeader => configuration
                .segment_processing
                .section_header
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Text => configuration
                .segment_processing
                .text
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::ListItem => configuration
                .segment_processing
                .list_item
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Table => configuration
                .segment_processing
                .table
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Picture => configuration
                .segment_processing
                .picture
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Caption => configuration
                .segment_processing
                .caption
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Formula => configuration
                .segment_processing
                .formula
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Footnote => configuration
                .segment_processing
                .footnote
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::PageHeader => configuration
                .segment_processing
                .page_header
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::PageFooter => configuration
                .segment_processing
                .page_footer
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
            SegmentType::Page => configuration
                .segment_processing
                .page
                .as_ref()
                .unwrap()
                .embed_sources
                .clone(),
        };

        let mut embed_parts = Vec::new();
        for source in &embed_sources {
            match source {
                EmbedSource::HTML => {
                    if !self.html.is_empty() {
                        embed_parts.push(self.html.clone());
                    }
                }
                EmbedSource::Markdown => {
                    if !self.markdown.is_empty() {
                        embed_parts.push(self.markdown.clone());
                    }
                }
                EmbedSource::LLM => {
                    if let Some(llm_content) = &self.llm {
                        if !llm_content.is_empty() {
                            embed_parts.push(llm_content.clone());
                        }
                    }
                }
                EmbedSource::Content => {
                    if !self.text.is_empty() {
                        embed_parts.push(self.text.clone());
                    }
                }
            }
        }

        embed_parts.join("\n")
    }

    fn count_with_huggingface_tokenizer(
        content: &str,
        model_name: &str,
    ) -> std::result::Result<u32, Box<dyn Error>> {
        let tokenizer = match Tokenizer::from_pretrained(model_name, None) {
            Ok(tokenizer) => tokenizer,
            Err(e) => return Err(e.to_string().into()),
        };

        let tokens = tokenizer.encode(content, true).map_err(|e| e.to_string())?;
        Ok(tokens.len() as u32)
    }

    pub fn count_embed_words(
        &self,
        configuration: &Configuration,
    ) -> std::result::Result<u32, Box<dyn Error>> {
        let cache_key = format!(
            "{}-{:?}",
            self.segment_id, configuration.chunk_processing.tokenizer
        );

        {
            if let Ok(mut cache) = WORD_COUNT_CACHE.lock() {
                if let Some(count) = cache.get(&cache_key) {
                    return Ok(*count);
                }
            }
        }

        let content = self.get_embed_content(configuration);

        let result: Result<u32, Box<dyn Error>> = match &configuration.chunk_processing.tokenizer {
            TokenizerType::Enum(tokenizer) => match tokenizer {
                crate::models::chunk_processing::Tokenizer::Word => {
                    // Simple whitespace tokenization
                    let count = content.split_whitespace().count();
                    Ok(count as u32)
                }
                crate::models::chunk_processing::Tokenizer::Cl100kBase => {
                    let bpe = cl100k_base().unwrap();
                    let tokens = bpe.encode_with_special_tokens(&content);
                    Ok(tokens.len() as u32)
                }
                _ => {
                    // For other enum tokenizers, use the HuggingFace tokenizer
                    let tokenizer_name = tokenizer.to_string();
                    Self::count_with_huggingface_tokenizer(&content, &tokenizer_name)
                }
            },
            TokenizerType::String(model_name) => {
                // Use the specified model name with the HuggingFace tokenizer
                Self::count_with_huggingface_tokenizer(&content, model_name)
            }
        };

        if let Ok(count) = result {
            if let Ok(mut cache) = WORD_COUNT_CACHE.lock() {
                cache.put(cache_key, count);
            }
        }

        result
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// Bounding box for an item. It is used for chunks, segments and OCR results.
pub struct BoundingBox {
    /// The left coordinate of the bounding box.
    pub left: f32,
    /// The top coordinate of the bounding box.
    pub top: f32,
    /// The width of the bounding box.
    pub width: f32,
    /// The height of the bounding box.
    pub height: f32,
}

impl BoundingBox {
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    fn intersects(&self, other: &BoundingBox) -> bool {
        if self.left + self.width < other.left || other.left + other.width < self.left {
            return false;
        }

        if self.top + self.height < other.top || other.top + other.height < self.top {
            return false;
        }

        true
    }

    pub fn intersection_area(&self, other: &BoundingBox) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let x_left = self.left.max(other.left);
        let x_right = (self.left + self.width).min(other.left + other.width);
        let y_top = self.top.max(other.top);
        let y_bottom = (self.top + self.height).min(other.top + other.height);

        (x_right - x_left) * (y_bottom - y_top)
    }

    pub fn scale(&mut self, scaling_factor: f32) {
        self.left *= scaling_factor;
        self.top *= scaling_factor;
        self.width *= scaling_factor;
        self.height *= scaling_factor;
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
/// OCR results for a segment
pub struct OCRResult {
    pub bbox: BoundingBox,
    /// The recognized text of the OCR result.
    pub text: String,
    /// The confidence score of the recognized text.
    pub confidence: Option<f32>,
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    PartialEq,
    EnumString,
    Display,
    ToSchema,
    ToSql,
    FromSql,
    Eq,
    Hash,
)]
/// All the possible types for a segment.
/// Note: Different configurations will produce different types.
/// Please refer to the documentation for more information.
pub enum SegmentType {
    Caption,
    Footnote,
    Formula,
    #[serde(alias = "List item")]
    ListItem,
    Page,
    #[serde(alias = "Page footer")]
    PageFooter,
    #[serde(alias = "Page header")]
    PageHeader,
    Picture,
    #[serde(alias = "Section header")]
    SectionHeader,
    Table,
    Text,
    Title,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::chunk_processing::{ChunkProcessing, Tokenizer, TokenizerType};
    use crate::models::llm::LlmProcessing;
    use crate::models::segment_processing::{EmbedSource, SegmentProcessing};
    use crate::models::upload::{ErrorHandlingStrategy, OcrStrategy, SegmentationStrategy};

    fn create_test_segment() -> Segment {
        Segment {
            bbox: BoundingBox::new(0.0, 0.0, 100.0, 100.0),
            confidence: Some(0.9),
            content: "This is content text".to_string(),
            html: "<p>This is HTML text</p>".to_string(),
            image: None,
            llm: Some("This is LLM text".to_string()),
            markdown: "This is *Markdown* text".to_string(),
            ocr: None,
            page_height: 1000.0,
            page_width: 800.0,
            page_number: 1,
            segment_id: "test-id".to_string(),
            segment_type: SegmentType::Table,
            text: "This is content text".to_string(),
        }
    }

    fn create_test_config() -> Configuration {
        let mut config = Configuration {
            chunk_processing: ChunkProcessing {
                ignore_headers_and_footers: true,
                target_length: 512,
                tokenizer: TokenizerType::Enum(Tokenizer::Cl100kBase),
            },
            expires_in: None,
            high_resolution: false,
            input_file_url: None,
            json_schema: None,
            model: None,
            ocr_strategy: OcrStrategy::All,
            segment_processing: SegmentProcessing::default(),
            segmentation_strategy: SegmentationStrategy::LayoutAnalysis,
            target_chunk_length: None,
            error_handling: ErrorHandlingStrategy::default(),
            llm_processing: LlmProcessing::default(),
            #[cfg(feature = "azure")]
            pipeline: None,
        };

        config
            .segment_processing
            .table
            .as_mut()
            .unwrap()
            .embed_sources = vec![EmbedSource::HTML, EmbedSource::Markdown];

        config
    }

    #[test]
    fn test_get_embed_content() {
        let segment = create_test_segment();
        let config = create_test_config();

        // Test with all sources enabled
        let content = segment.get_embed_content(&config);
        println!("Content: {content}");
        assert!(content.contains("This is HTML text"));
        assert!(content.contains("This is *Markdown* text"));
    }

    #[test]
    fn test_count_embed_words() {
        let segment = create_test_segment();
        let config = create_test_config();

        // When using the Word tokenizer, we should get the word count
        let word_count = segment.count_embed_words(&config).unwrap();
        println!("Word count: {word_count}");
        // The exact count will depend on the whitespace tokenizer, but it should be reasonable
        // Expected to be the sum of words from content, HTML, markdown, and LLM
        assert!(word_count > 0);
    }

    #[test]
    fn test_count_embed_words_with_many_tokenizers() {
        let segment = create_test_segment();
        let mut config = create_test_config();
        let identifiers = vec![
            TokenizerType::Enum(Tokenizer::Word),
            TokenizerType::Enum(Tokenizer::Cl100kBase),
            TokenizerType::Enum(Tokenizer::XlmRobertaBase),
            TokenizerType::Enum(Tokenizer::BertBaseUncased),
            TokenizerType::String("Qwen/Qwen-tokenizer".to_string()),
            TokenizerType::String("facebook/bart-large".to_string()),
        ];

        for identifier in identifiers {
            config.chunk_processing.tokenizer = identifier.clone();
            let word_count = segment.count_embed_words(&config).unwrap();
            println!("Word count for {identifier:?}: {word_count}");
            // The exact count will depend on the whitespace tokenizer, but it should be reasonable
            // Expected to be the sum of words from content, HTML, markdown, and LLM
            assert!(word_count > 0);
        }
    }
}

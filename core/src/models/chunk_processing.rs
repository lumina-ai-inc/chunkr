use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// Controls the setting for the chunking and post-processing of each chunk.
pub struct ChunkProcessing {
    #[serde(default = "default_ignore_headers_and_footers")]
    #[schema(value_type = bool, default = true)]
    /// Whether to ignore headers and footers in the chunking process.
    /// This is recommended as headers and footers break reading order across pages.
    pub ignore_headers_and_footers: bool,
    #[serde(default = "default_target_length")]
    #[schema(value_type = u32, default = 512)]
    /// The target number of words in each chunk. If 0, each chunk will contain a single segment.
    pub target_length: u32,
    /// The tokenizer to use for the chunking process.
    #[schema( value_type = TokenizerType, default = "Word")]
    #[serde(default)]
    pub tokenizer: TokenizerType,
}

impl ChunkProcessing {
    pub fn default() -> Self {
        Self {
            ignore_headers_and_footers: default_ignore_headers_and_footers(),
            target_length: default_target_length(),
            tokenizer: TokenizerType::Enum(Tokenizer::default()),
        }
    }
}

pub fn default_target_length() -> u32 {
    512
}

pub fn default_ignore_headers_and_footers() -> bool {
    true
}

#[derive(
    Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema, Default, Display, EnumString,
)]
/// Common tokenizers used for text processing.
///
/// These values represent standard tokenization approaches and popular pre-trained
/// tokenizers from the Hugging Face ecosystem.
pub enum Tokenizer {
    /// Split text by word boundaries
    #[default]
    Word,
    /// For OpenAI models (e.g. GPT-3.5, GPT-4, text-embedding-ada-002)
    Cl100kBase,
    /// For RoBERTa-based multilingual models
    #[strum(serialize = "xlm-roberta-base")]
    XlmRobertaBase,
    /// BERT base uncased tokenizer
    #[strum(serialize = "bert-base-uncased")]
    BertBaseUncased,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, Display)]
/// Specifies which tokenizer to use for the chunking process.
///
/// This type supports two ways of specifying a tokenizer:
/// 1. Using a predefined tokenizer from the `Tokenizer` enum
/// 2. Using any Hugging Face tokenizer by providing its model ID as a string
///    (e.g. "facebook/bart-large", "Qwen/Qwen-tokenizer", etc.)
///
/// When using a string, any valid Hugging Face tokenizer ID can be specified,
/// which will be loaded using the Hugging Face tokenizers library.
pub enum TokenizerType {
    /// Use one of the predefined tokenizer types
    Enum(Tokenizer),
    /// Use any Hugging Face tokenizer by specifying its model ID
    /// Examples: "Qwen/Qwen-tokenizer", "facebook/bart-large"
    String(String),
}

// Add Default implementation for TokenizerType
impl Default for TokenizerType {
    fn default() -> Self {
        TokenizerType::Enum(Tokenizer::default())
    }
}

// Manual implementation of ToSql and FromSql for TokenizerType
impl ToSql for TokenizerType {
    fn to_sql(
        &self,
        ty: &postgres_types::Type,
        out: &mut postgres_types::private::BytesMut,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        let s = match self {
            TokenizerType::Enum(t) => format!("enum:{}", t),
            TokenizerType::String(s) => format!("string:{}", s),
        };
        s.to_sql(ty, out)
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    postgres_types::to_sql_checked!();
}

impl<'a> FromSql<'a> for TokenizerType {
    fn from_sql(
        ty: &postgres_types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let s = String::from_sql(ty, raw)?;
        if s.starts_with("enum:") {
            let tokenizer_str = &s[5..];
            let tokenizer = tokenizer_str
                .parse::<Tokenizer>()
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            Ok(TokenizerType::Enum(tokenizer))
        } else if s.starts_with("string:") {
            Ok(TokenizerType::String(s[7..].to_string()))
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid TokenizerType format",
            )))
        }
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

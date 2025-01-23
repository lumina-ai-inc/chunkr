use crate::models::chunkr::search::ChunkContent;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct StructuredExtractionRequest {
    #[serde(flatten)]
    pub contents: Vec<ChunkContent>,
    pub structured_extraction: StructuredExtraction,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
#[postgres(name = "structured_extraction")]
pub struct StructuredExtraction {
    pub r#type: ExtractionType,
    pub json_schema: JsonSchema,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
pub struct JsonSchema {
    pub description: String,
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql, Default)]
#[postgres(name = "extraction_type")]
#[serde(rename_all = "snake_case")]
pub enum ExtractionType {
    #[default]
    JsonSchema,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
pub struct StructuredExtractionResponse {
    pub response: String,
}

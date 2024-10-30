use crate::models::server::segment::Chunk;
use crate::models::server::segment::Segment;
use crate::utils::services::embeddings::EmbeddingCache;
use crate::utils::services::llm::llm_call;
use crate::utils::services::search::search_embeddings;
use itertools::Itertools;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use tokio::task::JoinHandle;

/// Represents the structure of the incoming JSON schema.
#[derive(Debug, Deserialize)]
pub struct JsonSchema {
    pub title: String,
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: HashMap<String, Property>,
}

/// Represents each property within the JSON schema.
#[derive(Debug, Deserialize)]
pub struct Property {
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: Option<String>,
    pub default: Option<Value>,
}

/// Represents a field extracted from the JSON schema.
#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub description: String,
    pub field_type: String,
    pub default: Option<Value>,
}

/// Converts a JsonSchema into a vector of Fields.
impl JsonSchema {
    pub fn to_fields(&self) -> Vec<Field> {
        self.properties
            .iter()
            .map(|(name, prop)| Field {
                name: name.clone(),
                description: prop.description.clone().unwrap_or_default(),
                field_type: prop.prop_type.clone(),
                default: prop.default.clone(),
            })
            .collect()
    }
}

/// Performs structured extraction based on the provided JSON schema and chunks.
pub async fn perform_structured_extraction(
    json_schema: JsonSchema,
    chunks: Vec<Chunk>,
    embedding_url: String,
    embedding_key: String,
    llm_url: String,
    llm_key: String,
    top_k: usize,
    model_name: String,
) -> Result<Vec<(String, String)>, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let mut embedding_cache = EmbeddingCache::new();
    let mut results = Vec::new();

    let fields = json_schema.to_fields();

    let all_segments: Vec<Segment> = chunks.iter().flat_map(|c| c.segments.clone()).collect();
    let chunk_markdowns: Vec<String> = all_segments
        .iter()
        .filter_map(|s| s.markdown.clone())
        .collect();
    let segment_embeddings = embedding_cache
        .get_or_generate_embeddings(&client, &embedding_url, chunk_markdowns, top_k)
        .await?;

    let mut handles: Vec<JoinHandle<Result<(String, String), Box<dyn Error + Send + Sync>>>> =
        Vec::new();
    for field in fields {
        let client = client.clone();
        let embedding_url = embedding_url.clone();
        let llm_url = llm_url.clone();
        let llm_key = llm_key.clone();
        let model_name = model_name.clone();
        let mut embedding_cache = embedding_cache.clone();
        let segment_embeddings = segment_embeddings.clone();
        let all_segments = all_segments.clone();

        let handle = tokio::spawn(async move {
            let query = format!("{}: {}", field.name, field.description);

            let query_embedding = embedding_cache
                .get_or_generate_embeddings(&client, &embedding_url, vec![query.clone()], 1)
                .await?
                .get(0)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::Other, "Failed to get query embedding")
                })?
                .clone();

            let search_results =
                search_embeddings(&query_embedding, &all_segments, &segment_embeddings, top_k);
            let context = search_results
                .iter()
                .map(|res| res.segment.content.clone())
                .join("\n");

            let prompt = format!(
                "Field Name: {}\nField Description: {}\nField Type: {}\n\nContext:\n{}\n\nFind information for the field. Please respond directly and plainly with truthfulness according to the field type.",
                field.name, field.description, field.field_type, context
            );

            let extracted = llm_call(llm_url, llm_key, prompt, model_name, Some(8000), Some(0.0))
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok((field.name, extracted))
        });

        handles.push(handle);
    }

    for handle in handles {
        match handle.await? {
            Ok(result) => results.push(result),
            Err(e) => return Err(e),
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::server::segment::{BoundingBox, Segment, SegmentType};
    use crate::utils::configs::extraction_config::Config;
    use serde_json::json;
    use std::env;
    use tokio;

    #[tokio::test]
    async fn test_perform_structured_extraction() -> Result<(), Box<dyn Error + Send + Sync>> {
        // Setup test data
        let client = reqwest::Client::new();
        let embedding_url = "http://127.0.0.1:8085/embed".to_string();
        let config = Config::from_env().unwrap();
        let llm_url = config.ocr_llm_url;
        let llm_key = config.ocr_llm_key;
        let model_name =
            env::var("EXTRACTION__OCR_LLM_MODEL").unwrap_or_else(|_| "test_model".to_string());
        let top_k = 10;

        let segments = vec![
            Segment {
                segment_id: "1".to_string(),
                content: "Apple".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: Some("**Apple**".to_string()),
            },
            Segment {
                segment_id: "2".to_string(),
                content: "Orange".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: Some("_Orange_".to_string()),
            },
            Segment {
                segment_id: "3".to_string(),
                content: "Banana".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: Some("**Banana**".to_string()),
            },
            Segment {
                segment_id: "4".to_string(),
                content: "Carrot".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: Some("**Carrot**".to_string()),
            },
            Segment {
                segment_id: "5".to_string(),
                content: "Broccoli".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: Some("**Broccoli**".to_string()),
            },
        ];

        // Define JSON schema
        let json_schema = JsonSchema {
            title: "MainModel".to_string(),
            schema_type: "object".to_string(),
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "value1".to_string(),
                    Property {
                        title: Some("Value1".to_string()),
                        prop_type: "integer".to_string(),
                        description: Some("Description for value1".to_string()),
                        default: Some(json!(-1)),
                    },
                );
                props.insert(
                    "value2".to_string(),
                    Property {
                        title: Some("Value2".to_string()),
                        prop_type: "string".to_string(),
                        description: Some("Description for value2".to_string()),
                        default: Some(json!("default")),
                    },
                );
                props
            },
        };

        let chunks = vec![Chunk {
            segments: segments.clone(),
            chunk_length: 100,
        }];

        // Perform structured extraction
        let results = perform_structured_extraction(
            json_schema,
            chunks,
            embedding_url,
            "test_embedding_key".to_string(),
            llm_url,
            llm_key,
            top_k,
            model_name,
        )
        .await?;

        // Verify results
        assert!(!results.is_empty(), "Should return at least one result");
        assert_eq!(results[0].0, "value1"); // Based on the JSON schema keys

        Ok(())
    }
}

use crate::models::chunkr::open_ai::MessageContent;
use crate::models::chunkr::output::SegmentType;
use crate::models::chunkr::output::{Chunk, Segment};
use crate::models::chunkr::structured_extraction::{ExtractedField, ExtractedJson, JsonSchema};
use crate::utils::services::embeddings::EmbeddingCache;
use crate::utils::services::llm::{get_basic_message, open_ai_call};
use crate::utils::services::search::search_embeddings;

use itertools::Itertools;
use reqwest::Client;
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use tokio::task::JoinHandle;

pub async fn perform_structured_extraction(
    json_schema: JsonSchema,
    chunks: Vec<Chunk>,
    embedding_url: String,
    llm_url: String,
    llm_key: String,
    top_k: usize,
    model_name: String,
    batch_size: usize,
    content_type: String,
) -> Result<ExtractedJson, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let content_type_clone = content_type.clone();
    let fields = json_schema.to_fields();

    let all_segments: Vec<Segment> = chunks.iter().flat_map(|c| c.segments.clone()).collect();
    let chunk_markdowns: Vec<String> = all_segments
        .iter()
        .filter_map(|s| {
            if content_type == "markdown" {
                s.markdown
                    .clone()
                    .filter(|m| !m.is_empty())
                    .map(|m| format!("{} [{}]", m, s.segment_type))
            } else {
                Some(s.content.clone())
                    .filter(|c| !c.is_empty())
                    .map(|c| format!("{} [{}]", c, s.segment_type))
            }
        })
        .collect();
    let mut embedding_cache = EmbeddingCache {
        embeddings: HashMap::new(),
    };
    let segment_embeddings: Vec<Vec<f32>> = embedding_cache
        .get_or_generate_embeddings(&client, &embedding_url, chunk_markdowns, batch_size)
        .await?;
    let mut handles: Vec<
        JoinHandle<Result<(String, String, String), Box<dyn Error + Send + Sync>>>,
    > = Vec::new();
    for field in fields {
        let content_type_clone = content_type_clone.clone();
        let client = client.clone();
        let embedding_url = embedding_url.clone();
        let llm_url = llm_url.clone();
        let llm_key = llm_key.clone();
        let model_name = model_name.clone();
        let mut embedding_cache = embedding_cache.clone();
        let segment_embeddings = segment_embeddings.clone();
        let all_segments = all_segments.clone();
        let field_name = field.name.clone();
        let field_description = field.description.clone();
        let field_type = field.field_type.clone();
        let handle = tokio::spawn(async move {
            let query = format!("{}: {}", field_name, field_description);
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
                .filter_map(|s| {
                    if content_type_clone == "markdown" {
                        s.segment
                            .markdown
                            .clone()
                            .filter(|m| !m.is_empty())
                            .map(|m| format!("{} [{}]", m, s.segment.segment_type))
                    } else {
                        Some(s.segment.content.clone())
                            .filter(|c| !c.is_empty())
                            .map(|c| format!("{} [{}]", c, s.segment.segment_type))
                    }
                })
                .join("\n");

            let title_segment = if search_results
                .iter()
                .any(|s| s.segment.segment_type == SegmentType::Title)
            {
                search_results
                    .iter()
                    .find(|s| s.segment.segment_type == SegmentType::Title)
                    .map(|s| format!("\n\nTitle: {}", s.segment.content))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            let tag_instruction = match field_type.as_str() {
                "obj" | "object" | "dict" => "Output JSON within <json></json> tags.",
                "list" => "Output a list within <list></list> tags.",
                "string" => "Read the context and find what the user is asking for directly. Do not make up any information, and report truthfully what is in the document",
                _ => "Output the value appropriately. Be direct and to the point, no explanations. Just the required information in the type requested.",
            };

            let prompt = format!(
                    "Field Name: {}\nField Description: {}\nField Type: {}{} 
                    
                    CONTEXT: {}
                    
                    \n\nExtract the information for the field from the context provided. {} Ensure the output adheres to the schema without nesting.
                    You must accurately find the information for the field based on the name and description. Report the information in the type requested directly. Be intelligent. If the information you are looking for is not in the context provided, or is unrelated to the field, respond with a single <NA> tag. It is better to give a <NA> response rather than ambiously guessing.",
                    field_name, field_description, field_type, title_segment, context, tag_instruction,
                );

            let messages = get_basic_message(prompt).map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                )) as Box<dyn Error + Send + Sync>
            })?;

            let extracted = open_ai_call(
                llm_url,
                llm_key,
                model_name,
                messages,
                Some(8000),
                Some(0.0),
            )
            .await
            .map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                )) as Box<dyn Error + Send + Sync>
            })?;

            let content: String = match &extracted.choices.first().unwrap().message.content {
                MessageContent::String { content } => {
                    if content.trim() == "<NA>" {
                        String::new()
                    } else {
                        content.clone()
                    }
                }
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Invalid content type",
                    )
                    .into())
                }
            };
            Ok((field_name, field_type, content))
        });

        handles.push(handle);
    }

    let mut field_results = Vec::new();
    for handle in handles {
        match handle.await? {
            Ok(result) => field_results.push(result),
            Err(e) => return Err(e),
        }
    }

    let mut extracted_fields = Vec::new();
    for (name, field_type, value) in field_results {
        let parsed_value = match field_type.as_str() {
            "obj" => {
                let content = value
                    .split("<json>")
                    .nth(1)
                    .and_then(|s| s.split("</json>").next())
                    .unwrap_or(&value);
                serde_json::from_str(content.trim())?
            }
            "list" => {
                let content = value
                    .split("<list>")
                    .nth(1)
                    .and_then(|s| s.split("</list>").next())
                    .unwrap_or(&value);

                let list_items: Vec<&str> = content
                    .split(|c| c == ',' || c == '\n')
                    .map(|item| item.trim_matches(|c: char| c == '"' || c.is_whitespace()))
                    .filter(|item| !item.is_empty())
                    .collect();

                serde_json::Value::Array(
                    list_items
                        .into_iter()
                        .map(|item| serde_json::Value::String(item.to_string()))
                        .collect(),
                )
            }
            "int" => {
                let num = value.trim().parse::<i64>().unwrap_or(0);
                serde_json::Value::Number(num.into())
            }
            "float" => {
                let num = value.trim().parse::<f64>().unwrap_or(0.0);
                serde_json::Value::Number(
                    serde_json::Number::from_f64(num).unwrap_or(serde_json::Number::from(0)),
                )
            }
            "bool" => {
                let bool_val = value.trim().to_lowercase();
                serde_json::Value::Bool(bool_val == "true" || bool_val == "yes" || bool_val == "1")
            }
            "string" => serde_json::Value::String(value),
            _ => serde_json::Value::String(value),
        };
        let parsed_value = match parsed_value {
            serde_json::Value::String(s) => {
                let s = s.replace("<text>", "").replace("</text>", "");
                serde_json::Value::String(s)
            }
            _ => parsed_value,
        };
        extracted_fields.push(ExtractedField {
            name,
            field_type,
            value: parsed_value,
        });
    }
    Ok(ExtractedJson {
        title: json_schema.title,
        extracted_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::chunkr::output::{BoundingBox, Segment, SegmentType};
    use crate::models::chunkr::structured_extraction::Property;
    use crate::configs::llm_config::Config as LlmConfig;
    use crate::configs::worker_config::Config as WorkerConfig;
    use tokio;

    #[tokio::test]
    async fn st_extraction() -> Result<(), Box<dyn Error + Send + Sync>> {
        let embedding_url = "http://127.0.0.1:8085/embed".to_string();
        let worker_config = WorkerConfig::from_env().expect("Failed to load WorkerConfig");
        let llm_config = LlmConfig::from_env().expect("Failed to load LlmConfig");
        let llm_url = llm_config
            .structured_extraction_url
            .unwrap_or(llm_config.url);
        let llm_key = llm_config
            .structured_extraction_key
            .unwrap_or(llm_config.key);
        let model_name = llm_config
            .structured_extraction_model
            .unwrap_or(llm_config.model);
        let top_k = worker_config.structured_extraction_top_k;
        let batch_size = worker_config.structured_extraction_batch_size;
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
                markdown: Some("**Apple**: A sweet, edible fruit produced by an apple tree (Malus domestica). Rich in fiber and vitamin C.".to_string()),
            },
            Segment {
                segment_id: "2".to_string(),
                content: "Banana".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 100.0,
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
                markdown: Some("**Banana**: A long curved fruit with a thick yellow peel. High in potassium and carbohydrates.".to_string()),
            },
            Segment {
                segment_id: "3".to_string(),
                content: "Carrot".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 200.0,
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
                markdown: Some("**Carrot**: An orange root vegetable. Excellent source of beta carotene and fiber.".to_string()),
            },
            Segment {
                segment_id: "4".to_string(),
                content: "Broccoli".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 300.0,
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
                markdown: Some("**Broccoli**: A green vegetable with dense clusters of flower buds. High in vitamins C and K.".to_string()),
            },
            Segment {
                segment_id: "5".to_string(),
                content: "Orange".to_string(),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 400.0,
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
                markdown: Some("**Orange**: A citrus fruit with a bright orange peel. Excellent source of vitamin C.".to_string()),
            }
        ];

        let json_schema = JsonSchema {
            title: "Basket".to_string(),
            properties: vec![
                Property {
                    name: "fruits".to_string(),
                    title: Some("Fruits".to_string()),
                    prop_type: "list".to_string(),
                    description: Some("A list of fruits".to_string()),
                    default: None,
                },
                Property {
                    name: "greenest_vegetables".to_string(),
                    title: Some("greenest vegetables".to_string()),
                    prop_type: "string".to_string(),
                    description: Some("The greenest vegetable in the list".to_string()),
                    default: None,
                },
                Property {
                    name: "cars".to_string(),
                    title: Some("cars".to_string()),
                    prop_type: "list".to_string(),
                    description: Some("A list of cars".to_string()),
                    default: None,
                },
            ],
        };
        println!("JSON SCHEMA: {:?}", json_schema);
        let chunks = vec![Chunk {
            segments: segments.clone(),
            chunk_length: 100,
        }];

        let results = perform_structured_extraction(
            json_schema,
            chunks,
            embedding_url,
            llm_url,
            llm_key,
            top_k as usize,
            model_name,
            batch_size as usize,
            "content".to_string(),
        )
        .await?;

        println!("{:?}", results);
        assert!(
            !results.extracted_fields.is_empty(),
            "Should return at least one extracted field"
        );

        Ok(())
    }
}

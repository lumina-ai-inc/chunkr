use crate::configs::{
    llm_config::{get_prompt, Config as LlmConfig},
    search_config::Config as SearchConfig,
};
use crate::models::open_ai::MessageContent;
use crate::models::search::{Search, SearchResult};
use crate::models::structured_extraction::{
    StructuredExtractionRequest, StructuredExtractionResponse,
};
use crate::utils::services::llm::{create_basic_message, process_openai_request};
use futures::future::try_join_all;
use std::collections::HashMap;
use std::error::Error;

/// Extracts searchable strings from a JSON schema
fn extract_search_queries(schema: &serde_json::Value) -> Vec<String> {
    let mut queries = Vec::new();

    if let Some(obj) = schema.as_object() {
        let desc = obj.get("description").and_then(|d| d.as_str());
        let title = obj.get("title").and_then(|t| t.as_str());

        match (title, desc) {
            (Some(t), Some(d)) => queries.push(format!("{}: {}", t, d)),
            (Some(t), None) => queries.push(t.to_string()),
            (None, Some(d)) => queries.push(d.to_string()),
            (None, None) => {}
        }

        if let Some(props) = obj.get("properties").and_then(|p| p.as_object()) {
            for (_, value) in props {
                if let Some(prop_obj) = value.as_object() {
                    let desc = prop_obj.get("description").and_then(|d| d.as_str());
                    let title = prop_obj.get("title").and_then(|t| t.as_str());

                    match (title, desc) {
                        (Some(t), Some(d)) => queries.push(format!("{}: {}", t, d)),
                        (Some(t), None) => queries.push(t.to_string()),
                        (None, Some(d)) => queries.push(d.to_string()),
                        (None, None) => {}
                    }
                }
            }
        }

        if let Some(defs) = obj.get("$defs").and_then(|d| d.as_object()) {
            for (_, def_value) in defs {
                queries.extend(extract_search_queries(def_value));
            }
        }
    }

    queries
}

pub async fn perform_structured_extraction(
    structured_extraction_request: StructuredExtractionRequest,
) -> Result<StructuredExtractionResponse, Box<dyn Error>> {
    let llm_config = LlmConfig::from_env().unwrap();
    let search_config = SearchConfig::from_env().unwrap();
    let schema = &structured_extraction_request
        .structured_extraction
        .json_schema
        .schema;
    let search_queries = extract_search_queries(schema);
    let search = match Search::new(structured_extraction_request.contents).await {
        Ok(search) => search,
        Err(e) => return Err(e.to_string().into()),
    };
    let search_futures: Vec<_> = search_queries
        .iter()
        .map(|query| search.search(query))
        .collect();

    let search_results = match try_join_all(search_futures).await {
        Ok(search_results) => search_results,
        Err(e) => return Err(e.to_string().into()),
    };
    let mut seen_chunks = HashMap::new();
    for result in search_results.into_iter().flatten() {
        seen_chunks
            .entry(result.chunk.id.clone())
            .and_modify(|existing: &mut SearchResult| {
                if result.score > existing.score {
                    *existing = result.clone();
                }
            })
            .or_insert(result);
    }

    let mut final_results: Vec<_> = seen_chunks.values().cloned().collect();
    final_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    final_results.truncate(search_config.top_k);

    let mut user_values = HashMap::new();
    user_values.insert(
        "content".to_string(),
        final_results
            .iter()
            .map(|c| c.chunk.content.clone())
            .collect::<Vec<String>>()
            .join("\n"),
    );
    let system_message = create_basic_message(
        "system".to_string(),
        get_prompt("structured_extraction_system", &HashMap::new())?,
    )?;
    let user_message = create_basic_message(
        "user".to_string(),
        get_prompt("structured_extraction_user", &user_values)?,
    )?;
    println!("System message: {:?}", system_message);
    println!("User message: {:?}", user_message);
    let response = match process_openai_request(
        llm_config.url,
        llm_config.key,
        llm_config.model,
        vec![system_message, user_message],
        None,
        None,
        Some(structured_extraction_request.structured_extraction),
    )
    .await
    {
        Ok(response) => response,
        Err(e) => return Err(e.to_string().into()),
    };
    let response_text = match &response.choices[0].message.content {
        MessageContent::String { content } => content.clone(),
        _ => return Err("Invalid response format".into()),
    };
    Ok(StructuredExtractionResponse {
        response: response_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::search::{ChunkContent, SimpleChunk};
    use crate::models::structured_extraction::{
        ExtractionType, JsonSchema, StructuredExtraction,
    };
    use crate::utils::clients;
    use tokio;

    #[tokio::test]
    async fn test_structured_extraction() {
        clients::initialize().await;
        let json_schema = StructuredExtraction {
            r#type: ExtractionType::JsonSchema,
            json_schema: JsonSchema {
                description: "Fruit Facts".to_string(),
                name: "Fruit Facts".to_string(),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "fruit_name": {
                            "type": "string",
                            "description": "The name of the fruit"
                        },
                        "color": {
                            "type": "string",
                            "description": "The color of the fruit when ripe"
                        },
                        "calories_per_100g": {
                            "type": "number",
                            "description": "Number of calories per 100g serving"
                        },
                        "is_citrus": {
                            "type": "boolean",
                            "description": "Whether the fruit is a citrus fruit"
                        }
                    },
                    "required": ["fruit_name", "color", "calories_per_100g", "is_citrus"],
                    "additionalProperties": false
                }),
                strict: true,
            },
        };

        let contents = vec![
            SimpleChunk {
                id: uuid::Uuid::new_v4().to_string(),
                content: "Oranges are bright orange citrus fruits that contain about 47 calories per 100g serving.".to_string(),
            },
            SimpleChunk {
                id: uuid::Uuid::new_v4().to_string(),
                content: "Bananas have a yellow peel and are not citrus fruits. They contain approximately 89 calories per 100g.".to_string(),
            },
            SimpleChunk {
                id: uuid::Uuid::new_v4().to_string(),
                content: "Lemons are yellow citrus fruits with around 29 calories per 100g.".to_string(),
            },
        ];

        let response = perform_structured_extraction(StructuredExtractionRequest {
            contents: contents
                .iter()
                .map(|c| ChunkContent::Simple(c.clone()))
                .collect(),
            structured_extraction: json_schema,
        })
        .await;
        println!("Response: {:?}", response);
        assert!(response.is_ok());
    }
}

use crate::models::server::segment::{Chunk, Segment};
use crate::utils::services::embeddings::EmbeddingCache;
use crate::utils::services::llm::llm_call;
use crate::utils::services::search::search_embeddings;
use bytes::BytesMut;
use itertools::Itertools;
use postgres_types::{FromSql, IsNull, ToSql, Type};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use tokio::task::JoinHandle;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct JsonSchema {
    pub title: String,
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: Vec<Property>,
}

impl std::str::FromStr for JsonSchema {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct Property {
    pub name: String,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: Option<String>,
    pub default: Option<String>,
}

#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub description: String,
    pub field_type: String,
    pub default: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct ExtractedJson {
    pub title: String,
    pub schema_type: String,
    pub extracted_fields: Vec<ExtractedField>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ExtractedField {
    pub name: String,
    pub field_type: String,
    #[serde(
        serialize_with = "serialize_value",
        deserialize_with = "deserialize_value"
    )]
    pub value: serde_json::Value,
}

impl ToSql for ExtractedField {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        let json_str = serde_json::to_string(&self.value)?;
        json_str.to_sql(ty, out)
    }

    fn accepts(ty: &Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    fn to_sql_checked(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        let json_str = serde_json::to_string(&self.value)?;
        json_str.to_sql_checked(ty, out)
    }
}

impl<'a> FromSql<'a> for ExtractedField {
    fn from_sql(ty: &Type, raw: &'a [u8]) -> Result<Self, Box<dyn Error + Sync + Send>> {
        let json_str = <String as FromSql>::from_sql(ty, raw)?;
        let value = serde_json::from_str(&json_str)?;
        Ok(ExtractedField {
            name: String::new(),
            field_type: String::new(),
            value,
        })
    }

    fn accepts(ty: &Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

fn serialize_value<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    value.serialize(serializer)
}

fn deserialize_value<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer)
}

impl JsonSchema {
    pub fn to_fields(&self) -> Vec<Field> {
        self.properties
            .iter()
            .map(|prop| Field {
                name: prop.name.clone(),
                description: prop.description.clone().unwrap_or_default(),
                field_type: prop.prop_type.clone(),
                default: prop.default.clone(),
            })
            .collect()
    }
}

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
        .filter_map(|s| if content_type == "markdown" {
            s.markdown.clone()
        } else {
            Some(s.content.clone())
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
                .map(|res| if content_type_clone == "markdown" {
                    res.segment.markdown.clone().unwrap_or_default()
                } else {
                    res.segment.content.clone()
                })
                .join("\n");

            let tag_instruction = match field_type.as_str() {
                "obj" | "object" | "dict" => "Output JSON within <json></json> tags.",
                "list" => "Output a list within <list></list> tags.",
                "string" => "Read the context and find what the user is asking for directly. Do not make up any information, and report truthfully what is in the document",
                _ => "Output the value appropriately. Be direct and to the point, no explanations. Just the required information in the type requested.",
            };

            let prompt = format!(
                    "Field Name: {}\nField Description: {}\nField Type: {}\n\nContext:\n{}\n\nExtract the information for the field. {} Ensure the output adheres to the schema without nesting.
                    You must accurately find the information for the field based on the name and description. Report the information in the type requested directly.",
                    field_name, field_description, field_type, context, tag_instruction,
                );

            let extracted = llm_call(llm_url, llm_key, prompt, model_name, Some(8000), Some(0.0))
                .await
                .map_err(|e| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    )) as Box<dyn Error + Send + Sync>
                })?;
            Ok((field_name, field_type, extracted))
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
        schema_type: json_schema.schema_type,
        extracted_fields,
    })
}

use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, Distance, PointStruct, SearchPoints, VectorParams, VectorsConfig,
};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::OnceCell;

static QDRANT_CLIENT: OnceCell<Arc<QdrantClient>> = OnceCell::const_new();

/// Initialize Qdrant client
pub async fn get_qdrant_client() -> Result<Arc<QdrantClient>, Box<dyn Error + Send + Sync>> {
    QDRANT_CLIENT
        .get_or_try_init(|| async {
            let qdrant_url = std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string());
            
            let mut config = QdrantClientConfig::from_url(&qdrant_url);
            
            // Add API key if provided
            if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
                config.set_api_key(&api_key);
            }
            
            let client = QdrantClient::new(Some(config))?;
            Ok(Arc::new(client))
        })
        .await
        .map(|client| client.clone())
}

pub struct VectorDB {
    client: Arc<QdrantClient>,
}

impl VectorDB {
    pub async fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = get_qdrant_client().await?;
        Ok(Self { client })
    }

    /// Initialize all collections needed for the application
    pub async fn initialize_collections(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create documents collection
        self.create_collection_if_not_exists("documents", 1536, Distance::Cosine)
            .await?;

        // Create facts collection with filtering
        self.create_collection_if_not_exists("facts", 1536, Distance::Cosine)
            .await?;

        // Create conversations collection
        self.create_collection_if_not_exists("conversations", 1536, Distance::Cosine)
            .await?;

        println!("Vector DB collections initialized successfully");
        Ok(())
    }

    /// Create a collection if it doesn't exist
    async fn create_collection_if_not_exists(
        &self,
        collection_name: &str,
        vector_size: u64,
        distance: Distance,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Check if collection exists
        let collections = self.client.list_collections().await?;
        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !exists {
            println!("Creating collection: {}", collection_name);
            self.client
                .create_collection(&CreateCollection {
                    collection_name: collection_name.to_string(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: vector_size,
                            distance: distance.into(),
                            ..Default::default()
                        })),
                    }),
                    ..Default::default()
                })
                .await?;
            println!("Collection created: {}", collection_name);
        } else {
            println!("Collection already exists: {}", collection_name);
        }

        Ok(())
    }

    /// Upsert a single embedding into a collection
    pub async fn upsert_embedding(
        &self,
        collection: &str,
        id: String,
        vector: Vec<f32>,
        payload: HashMap<String, JsonValue>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let payload_map: HashMap<String, Value> = payload
            .into_iter()
            .map(|(k, v)| (k, json_to_qdrant_value(v)))
            .collect();

        let point = PointStruct::new(id, vector, payload_map);

        self.client
            .upsert_points_blocking(collection, None, vec![point], None)
            .await?;

        Ok(())
    }

    /// Upsert multiple embeddings into a collection
    pub async fn upsert_embeddings_batch(
        &self,
        collection: &str,
        points: Vec<(String, Vec<f32>, HashMap<String, JsonValue>)>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let qdrant_points: Vec<PointStruct> = points
            .into_iter()
            .map(|(id, vector, payload)| {
                let payload_map: HashMap<String, Value> = payload
                    .into_iter()
                    .map(|(k, v)| (k, json_to_qdrant_value(v)))
                    .collect();
                PointStruct::new(id, vector, payload_map)
            })
            .collect();

        self.client
            .upsert_points_blocking(collection, None, qdrant_points, None)
            .await?;

        Ok(())
    }

    /// Search for similar vectors
    pub async fn search_similar(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        limit: u64,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<SearchResult>, Box<dyn Error + Send + Sync>> {
        let search_result = self
            .client
            .search_points(&SearchPoints {
                collection_name: collection.to_string(),
                vector: query_vector,
                filter,
                limit,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await?;

        let results: Vec<SearchResult> = search_result
            .result
            .into_iter()
            .map(|point| SearchResult {
                id: point.id.map(|id| match id.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => {
                        num.to_string()
                    }
                    None => String::new(),
                }),
                score: point.score,
                payload: point
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, qdrant_value_to_json(v)))
                    .collect(),
            })
            .collect();

        Ok(results)
    }

    /// Delete points by filter
    pub async fn delete_by_filter(
        &self,
        collection: &str,
        filter: qdrant_client::qdrant::Filter,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.client
            .delete_points(collection, None, &filter.into(), None)
            .await?;
        Ok(())
    }

    /// Delete a collection
    pub async fn delete_collection(
        &self,
        collection: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.client.delete_collection(collection).await?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: Option<String>,
    pub score: f32,
    pub payload: HashMap<String, JsonValue>,
}

/// Convert serde_json::Value to qdrant Value
fn json_to_qdrant_value(json: JsonValue) -> Value {
    use qdrant_client::qdrant::value::Kind;
    
    match json {
        JsonValue::Null => Value {
            kind: Some(Kind::NullValue(0)),
        },
        JsonValue::Bool(b) => Value {
            kind: Some(Kind::BoolValue(b)),
        },
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value {
                    kind: Some(Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                Value {
                    kind: Some(Kind::DoubleValue(f)),
                }
            } else {
                Value {
                    kind: Some(Kind::NullValue(0)),
                }
            }
        }
        JsonValue::String(s) => Value {
            kind: Some(Kind::StringValue(s)),
        },
        JsonValue::Array(arr) => {
            let values: Vec<Value> = arr.into_iter().map(json_to_qdrant_value).collect();
            Value {
                kind: Some(Kind::ListValue(qdrant_client::qdrant::ListValue { values })),
            }
        }
        JsonValue::Object(obj) => {
            let map: HashMap<String, Value> = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_qdrant_value(v)))
                .collect();
            Value {
                kind: Some(Kind::StructValue(qdrant_client::qdrant::Struct { fields: map })),
            }
        }
    }
}

/// Convert qdrant Value to serde_json::Value
fn qdrant_value_to_json(value: Value) -> JsonValue {
    match value.kind {
        Some(qdrant_client::qdrant::value::Kind::NullValue(_)) => JsonValue::Null,
        Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => JsonValue::Bool(b),
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
            JsonValue::Number(i.into())
        }
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)) => {
            JsonValue::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)))
        }
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => JsonValue::String(s),
        Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => {
            let arr: Vec<JsonValue> = list.values.into_iter().map(qdrant_value_to_json).collect();
            JsonValue::Array(arr)
        }
        Some(qdrant_client::qdrant::value::Kind::StructValue(s)) => {
            let obj: serde_json::Map<String, JsonValue> = s
                .fields
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect();
            JsonValue::Object(obj)
        }
        None => JsonValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Qdrant running
    async fn test_vector_db_operations() {
        let vector_db = VectorDB::new().await.unwrap();
        
        // Initialize collections
        vector_db.initialize_collections().await.unwrap();
        
        // Test upsert
        let mut payload = HashMap::new();
        payload.insert("text".to_string(), JsonValue::String("test document".to_string()));
        payload.insert("deal_id".to_string(), JsonValue::String("deal-123".to_string()));
        
        let vector = vec![0.1; 1536]; // Dummy embedding
        
        vector_db
            .upsert_embedding("documents", "doc-1".to_string(), vector.clone(), payload)
            .await
            .unwrap();
        
        // Test search
        let results = vector_db
            .search_similar("documents", vector, 5, None)
            .await
            .unwrap();
        
        assert!(!results.is_empty());
    }
}


use crate::models::server::segment::SegmentType;
use crate::utils::configs::extraction_config::Config;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct LLMConfig {
    pub model: LLMModel,
    pub temperature: f32,
    pub max_tokens: usize,
    pub affected_segments: Vec<SegmentType>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub enum LLMModel {
    GPT4o,
    GPT4oMini,
    Haiku,
    Sonnet3_5,
    Qwen2VL,
}

impl LLMModel {
    pub fn base_url<'a>(&self, config: &'a Config) -> Option<&'a str> {
        match self {
            // ... other cases ...
            LLMModel::Qwen2VL => config.qwen_url.as_deref(),
            _ => None,
        }
    }
}

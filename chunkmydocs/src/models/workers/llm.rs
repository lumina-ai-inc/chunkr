use crate::models::server::segment::SegmentType;
use postgres_types::{ FromSql, ToSql };
use serde::{ Deserialize, Serialize };
use serde_yaml;

#[derive(Debug, Serialize, Deserialize, FromSql, ToSql, Clone, PartialEq)]
#[postgres(name = "llm_model")]
pub enum LLMModel {
    #[postgres(name = "qwen_2_vl_7b_instruct")]
    Qwen2VL7BInstruct,
    #[postgres(name = "gemini_flash_1_5_8b")]
    GeminiFlash158B,
    #[postgres(name = "llama_3_2_11b_vision_instruct")]
    LLama3211BVisionInstruct,
    #[postgres(name = "claude_3_haiku")]
    Claude3Haiku,
    #[postgres(name = "gpt_4o")]
    GPT4O,
    #[postgres(name = "gpt_4o_mini")]
    GPT4OMini,
}

impl LLMModel {

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "qwen/qwen-2-vl-7b-instruct" => Ok(LLMModel::Qwen2VL7BInstruct),
            "google/gemini-flash-1.5-8b" => Ok(LLMModel::GeminiFlash158B),
            "meta-llama/llama-3.2-11b-vision-instruct" => Ok(LLMModel::LLama3211BVisionInstruct),
            "anthropic/claude-3-haiku" => Ok(LLMModel::Claude3Haiku),
            "openai/gpt-4o" => Ok(LLMModel::GPT4O),
            "openai/gpt-4o-mini" => Ok(LLMModel::GPT4OMini),
            _ => Err(format!("Invalid LLM model: {}", s)),
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            LLMModel::Qwen2VL7BInstruct => "qwen/qwen-2-vl-7b-instruct",
            LLMModel::GeminiFlash158B => "google/gemini-flash-1.5-8b",
            LLMModel::LLama3211BVisionInstruct => "meta-llama/llama-3.2-11b-vision-instruct",
            LLMModel::Claude3Haiku => "anthropic/claude-3-haiku",
            LLMModel::GPT4O => "openai/gpt-4o",
            LLMModel::GPT4OMini => "openai/gpt-4o-mini",
        }
    }

    fn load_prompt(&self, prompt_key: &str) -> String {
        let path = format!("../../utils/prompts/2024-10-23.yaml");
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let yaml: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or_default();
        yaml[prompt_key].as_str().unwrap_or_default().to_string()
    }

    fn get_prompt_key(&self, segment_type: &SegmentType) -> String {
        match segment_type {
            SegmentType::Table => "table_to_html".to_string(),
            _ => "default".to_string(),
        }
    }

    pub fn get_prompt(&self, segment_type: &SegmentType) -> String {
        self.load_prompt(&self.get_prompt_key(segment_type))
    }
}


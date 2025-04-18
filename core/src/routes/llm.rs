use crate::configs::llm_config::{Config, LlmModelPublic};
use actix_web::{Error, HttpResponse};

pub async fn get_models_ids() -> Result<HttpResponse, Error> {
    let llm_config = Config::from_env().unwrap();
    let models: Vec<LlmModelPublic> = llm_config
        .llm_models
        .unwrap()
        .iter()
        .map(|m| LlmModelPublic::from(m.clone()))
        .collect();
    Ok(HttpResponse::Ok().json(models))
}

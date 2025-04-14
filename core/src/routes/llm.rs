use crate::configs::llm_config::Config;
use actix_web::{Error, HttpResponse};

pub async fn get_models_ids() -> Result<HttpResponse, Error> {
    let llm_config = Config::from_env().unwrap();
    let models_ids: Vec<String> = llm_config.llm_models.unwrap().iter().map(|m| m.id.clone()).collect();
    Ok(HttpResponse::Ok().json(models_ids))
}

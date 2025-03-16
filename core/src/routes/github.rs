use crate::configs::github_config::Config;
use actix_web::HttpResponse;
use reqwest::Client;
use serde_json::Value;

pub async fn get_github_repo_info() -> Result<HttpResponse, Box<dyn std::error::Error>> {
    let config = Config::from_env().unwrap();
    let client = Client::new();
    let url = "https://api.github.com/repos/lumina-ai-inc/chunkr".to_string();
    let mut request = client
        .get(url)
        .header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        );

    if let Some(github_token) = config.github_token {
        request = request.header("Authorization", format!("Bearer {}", github_token));
    }

    let response = request.send().await?.error_for_status()?;

    let repo_info = response.json::<Value>().await?;
    Ok(HttpResponse::Ok().json(repo_info))
}

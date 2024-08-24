use reqwest::multipart;
use reqwest::Client;
use serde_json::json;
use std::error::Error;

pub async fn upload_to_s3(
    client: Client,
    s3_path: &str,
    file_name: &str,
    buffer: Vec<u8>,
    expiration: Option<String>,
) -> Result<bool, Box<dyn Error>> {
    let metadata = json!({
        "location": format!("s3://{}", s3_path),
        "expiration": expiration
    });

    let form = multipart::Form::new()
        .part(
            "metadata",
            multipart::Part::text(serde_json::to_string(&metadata)?)
                .mime_str("application/json")?,
        )
        .part(
            "file",
            multipart::Part::bytes(buffer)
                .file_name(file_name.to_string())
                .mime_str("application/pdf")?,
        );

    let response = client
        .post("https://storage.lumina.sh/upload")
        .multipart(form)
        .send()
        .await?;

    println!("Response: {:?}", response);
    Ok(response.status().is_success())
}

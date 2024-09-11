use crate::models::server::extract::{TableOcr, TableOcrModel};
use crate::utils::configs::extraction_config::Config;
use reqwest::{multipart, Client as ReqwestClient};
use std::{fs, path::Path};
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

async fn call_table_extraction_api(
    url: &str,
    file_path: &Path,
    ocr_model: TableOcrModel,
    output_format: TableOcr,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;

    let file_name = file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();
    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);

    let mut form = multipart::Form::new().part("file", part);

    // Add OCR model to form
    form = form.text("ocr_model", ocr_model.to_string().to_lowercase());

    let endpoint = if ocr_model == TableOcrModel::Qwen {
        "/table/json"
    } else {
        match output_format {
            TableOcr::HTML => "/table/html",
            TableOcr::JSON => "/table/json",
        }
    };

    let response = if ocr_model == TableOcrModel::Qwen {
        let prompt = "Return the provided complex table in JSON format that preserves information and hierarchy from the table at 100 percent accuracy.
                            Create Intelligent lists and nesting to preserve the format of the table.";
        form = form.text("prompt", prompt);
        client
            .post(url)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?
    } else {
        client
            .post(format!("{}{}", url, endpoint))
            .multipart(form)
            .send()
            .await?
            .error_for_status()?
    };
    Ok(response.text().await?)
}

pub async fn table_extraction_from_image(
    file_path: &Path,
    ocr_model: TableOcrModel,
    output_format: TableOcr,
) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let mut url = config.table_ocr_url.ok_or("Table OCR URL not configured")?;

    let output = call_table_extraction_api(&url, file_path, ocr_model, output_format).await?;

    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[tokio::test]
    async fn table_extraction() -> Result<(), Box<dyn std::error::Error>> {
        let input_file = Path::new("input/test.png");
        let ocr_model = TableOcrModel::EasyOcr;

        // Test JSON output
        let json_output =
            table_extraction_from_image(input_file, ocr_model, TableOcr::JSON).await?;
        println!("JSON output: {}", json_output);
        fs::write(Path::new("output/test_output.json"), json_output)?;

        // Test HTML output
        let html_output =
            table_extraction_from_image(input_file, ocr_model, TableOcr::HTML).await?;
        println!("HTML output: {}", html_output);

        fs::write(Path::new("output/test_output.html"), html_output)?;

        Ok(())
    }
}

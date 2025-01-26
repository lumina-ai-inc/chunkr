use crate::utils::clients;
use aws_sdk_s3::{presigning::PresigningConfig, primitives::ByteStream};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use once_cell::sync::Lazy;
use regex::Regex;
use std::io::copy;
use std::path::Path;
use std::time::Duration;
use tempfile::NamedTempFile;

static S3_PATH_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^s3://[a-zA-Z0-9.\-_]{3,63}/.*$").unwrap());

pub fn extract_bucket_and_key(
    s3_path: &str,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = s3_path.trim_start_matches("s3://").splitn(2, '/').collect();
    match parts.len() != 2 {
        true => Err("Invalid S3 path format".into()),
        false => Ok((parts[0].to_string(), parts[1].to_string())),
    }
}

pub fn validate_s3_path(s3_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    match S3_PATH_REGEX.is_match(s3_path) {
        true => Ok(()),
        false => Err("Invalid S3 path format".into()),
    }
}

pub async fn generate_presigned_url(
    location: &str,
    external: bool,
    expires_in: Option<Duration>,
    base64_urls: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let s3_client = if external {
        clients::get_external_s3_client()
    } else {
        clients::get_s3_client()
    };
    let (bucket, key) = extract_bucket_and_key(location)?;

    if base64_urls {
        let output = s3_client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await?;
        let image_bytes = output.body.collect().await?.into_bytes();
        let base64_string = STANDARD.encode(&image_bytes);
        return Ok(base64_string);
    }

    let expiration = expires_in.unwrap_or(Duration::from_secs(3600));
    let mut get_object = s3_client.get_object().bucket(bucket).key(key);
    get_object = get_object
        .response_content_disposition("inline")
        .response_content_encoding("utf-8");
    let presigned_request = get_object
        .presigned(PresigningConfig::expires_in(expiration)?)
        .await?;
    Ok(presigned_request.uri().to_string())
}

pub async fn upload_to_s3_from_memory(
    s3_path: &str,
    content: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let s3_client = clients::get_s3_client();
    let (bucket, key) = parse_s3_path(s3_path)?;
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(ByteStream::from(Bytes::from(content.to_vec())))
        .send()
        .await?;
    Ok(())
}

// Helper function to parse S3 path
fn parse_s3_path(s3_path: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = s3_path.trim_start_matches("s3://").split('/').collect();
    if parts.len() < 2 {
        return Err("Invalid S3 path".into());
    }
    Ok((parts[0].to_string(), parts[1..].join("/")))
}

pub async fn upload_to_s3(
    s3_location: &str,
    file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let s3_client = clients::get_s3_client();
    let file_content = tokio::fs::read(file_path).await?;
    let (bucket, key) = extract_bucket_and_key(s3_location)?;
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key.to_string())
        .body(ByteStream::from(file_content))
        .send()
        .await?;

    Ok(())
}

pub async fn download_to_tempfile(
    location: &str,
    expires_in: Option<Duration>,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let reqwest_client = clients::get_reqwest_client();
    let unsigned_url = generate_presigned_url(location, false, expires_in, false).await?;
    let mut temp_file = NamedTempFile::new()?;
    let content = reqwest_client
        .get(&unsigned_url)
        .send()
        .await?
        .bytes()
        .await?;
    copy(&mut content.as_ref(), &mut temp_file)?;
    Ok(temp_file)
}

pub async fn download_to_given_tempfile(
    mut temp_file: &NamedTempFile,
    location: &str,
    expires_in: Option<Duration>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reqwest_client = clients::get_reqwest_client();
    let unsigned_url = generate_presigned_url(location, false, expires_in, false).await?;
    let content = reqwest_client
        .get(&unsigned_url)
        .send()
        .await?
        .bytes()
        .await?;
    copy(&mut content.as_ref(), &mut temp_file)?;

    Ok(())
}

pub async fn delete_folder(location: &str) -> Result<(), Box<dyn std::error::Error>> {
    let s3_client = clients::get_s3_client();
    let (bucket, prefix) = extract_bucket_and_key(location)?;
    let objects = s3_client
        .list_objects_v2()
        .bucket(&bucket)
        .prefix(&prefix)
        .send()
        .await?;

    let futures = objects.contents().iter().map(|object| {
        let bucket = bucket.clone();
        let key = object.key().unwrap().to_string();
        let s3_client = s3_client.clone();
        async move {
            s3_client
                .delete_object()
                .bucket(bucket)
                .key(key)
                .send()
                .await
        }
    });

    let results = futures::future::join_all(futures).await;
    for result in results {
        match result {
            Ok(_) => (),
            Err(e) => println!("Error deleting object: {:?}", e),
        }
    }
    Ok(())
}

use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::operation::head_object::HeadObjectError;
use aws_sdk_s3::{presigning::PresigningConfig, primitives::ByteStream, Client as S3Client};
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
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
    s3_client: &S3Client,
    location: &str,
    expires_in: Option<Duration>,
) -> Result<String, Box<dyn std::error::Error>> {
    let expiration = expires_in.unwrap_or(Duration::from_secs(3600));

    // Extract bucket and key from the S3 path
    let (bucket, key) = extract_bucket_and_key(location)?;

    // Generate presigned URL
    let presigned_request = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .presigned(PresigningConfig::expires_in(expiration)?)
        .await?;

    Ok(presigned_request.uri().to_string())
}

pub async fn generate_presigned_url_if_exists(
    s3_client: &S3Client,
    location: &str,
    expires_in: Option<Duration>,
) -> Result<String, Box<dyn std::error::Error>> {
    let expiration = expires_in.unwrap_or(Duration::from_secs(3600));

    let (bucket, key) = extract_bucket_and_key(location)?;

    match s3_client
        .head_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
    {
        Ok(_) => {
            let presigned_request = s3_client
                .get_object()
                .bucket(bucket)
                .key(key)
                .presigned(PresigningConfig::expires_in(expiration)?)
                .await?;

            Ok(presigned_request.uri().to_string())
        }
        Err(SdkError::ServiceError(err)) if matches!(err.err(), HeadObjectError::NotFound(_)) => {
            Err(format!("Object does not exist: {}", location).into())
        }
        Err(err) => Err(format!("Error checking object existence: {}", err).into()),
    }
}
use bytes::Bytes;

pub async fn upload_to_s3_from_memory(
    s3_client: &S3Client,
    s3_path: &str,
    content: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
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
    s3_client: &S3Client,
    s3_location: &str,
    file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
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
    s3_client: &S3Client,
    reqwest_client: &Client,
    location: &str,
    expires_in: Option<Duration>,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let unsigned_url = generate_presigned_url(s3_client, location, expires_in).await?;

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

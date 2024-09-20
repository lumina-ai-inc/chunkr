use crate::utils::configs::extraction_config::Config;
use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;
use reqwest::{multipart, Client as ReqwestClient};
use std::{
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;
use tokio::sync::OnceCell;
use tokio::time::{sleep, Duration};

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

async fn call_grobid(
    config: &Config,
    file_path: &Path,
) -> Result<reqwest::Response, reqwest::Error> {
    println!("Calling grobid for {}", file_path.display());
    let url: String = format!("{}/api/processFulltextDocument", config.grobid_url.clone().unwrap());
    let client = get_reqwest_client().await;

    let file_name = file_path.file_name().unwrap().to_str().unwrap().to_string();
    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);

    let form = multipart::Form::new()
        .part("input", part)
        .text("consolidateHeader", "0")
        .text("consolidateCitations", "0")
        .text("consolidateFunders", "0")
        .text("includeRawCitations", "0")
        .text("includeRawAffiliations", "0")
        .text("segmentSentences", "1")
        .text("generateIds", "1")
        .text("teiCoordinates", "persName")
        .text("teiCoordinates", "figure")
        .text("teiCoordinates", "ref")
        .text("teiCoordinates", "biblStruct")
        .text("teiCoordinates", "formula")
        .text("teiCoordinates", "p")
        .text("teiCoordinates", "s")
        .text("teiCoordinates", "note")
        .text("teiCoordinates", "head")
        .text("teiCoordinates", "title");

    client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()
}

fn clean_xml(xml: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut writer = Writer::new(Cursor::new(Vec::new()));
    let mut buf = Vec::new();
    let mut skip_depth = 0;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"encodingDesc" => {
                skip_depth += 1;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"encodingDesc" => {
                skip_depth -= 1;
                if skip_depth == 0 {
                    buf.clear();
                    continue;
                }
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"TEI" => {
                let mut elem = BytesStart::new("TEI");
                for attr in e.attributes().filter_map(Result::ok) {
                    if attr.key.as_ref() != b"xsi:schemaLocation" {
                        elem.push_attribute(attr);
                    }
                }
                writer.write_event(Event::Start(elem))?;
            }
            Ok(Event::Eof) => break,
            Ok(event) => {
                if skip_depth == 0 {
                    writer.write_event(event)?;
                }
            }
            Err(e) => return Err(Box::new(e)),
        }
        buf.clear();
    }

    let result = writer.into_inner().into_inner();
    println!("Removed encodingDesc and xsi:schemaLocation");
    Ok(String::from_utf8(result)?)
}

async fn handle_grobid_requests(file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    loop {
        match call_grobid(&config, file_path).await {
            Ok(response) => {
                println!("Grobid response: {:?}", response.status());
                if response.status() != reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    return Ok(response.text().await?);
                }
            }
            Err(e) => {
                if e.status() != Some(reqwest::StatusCode::SERVICE_UNAVAILABLE) {
                    return Err(Box::new(e));
                }
            }
        }
        println!("Received 503 status, retrying in 1 second...");
        sleep(Duration::from_secs(1)).await;
    }
}

pub async fn grobid_extraction(file_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let xml = handle_grobid_requests(file_path).await?;
    let clean_xml = clean_xml(&xml)?;
    let mut output_temp_file = NamedTempFile::new()?;
    output_temp_file.write_all(clean_xml.as_bytes())?;

    Ok(output_temp_file.into_temp_path().keep()?.to_path_buf())
}

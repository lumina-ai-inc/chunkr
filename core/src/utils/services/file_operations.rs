use std::error::Error;
use std::process::Command;
use tempfile::NamedTempFile;

pub fn check_file_type(file: &NamedTempFile) -> Result<String, Box<dyn Error>> {
    let output = Command::new("file")
        .arg("--mime-type")
        .arg("-b")
        .arg(file.path().to_str().unwrap())
        .output()?;

    let mime_type = String::from_utf8(output.stdout)?.trim().to_string();

    match mime_type.as_str() {
        "application/pdf"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.ms-powerpoint"
        | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        | "application/vnd.ms-excel"
        | "image/jpeg"
        | "image/png"
        | "image/jpg" => Ok(mime_type),
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Unsupported file type: {}", mime_type),
        ))),
    }
}

pub fn convert_to_pdf(input_file: &NamedTempFile) -> Result<NamedTempFile, Box<dyn Error>> {
    let output_dir = input_file.path().parent().unwrap();

    let output = Command::new("libreoffice")
        .args(&[
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir.to_str().unwrap(),
            input_file.path().to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("LibreOffice conversion failed: {:?}", output),
        )));
    }

    let pdf_file_name = input_file
        .path()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
        + ".pdf";

    let pdf_file_path = output_dir.join(pdf_file_name);

    if pdf_file_path.exists() {
        let temp_file = NamedTempFile::new()?;
        std::fs::copy(&pdf_file_path, temp_file.path())?;
        Ok(temp_file)
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Converted PDF file not found in output directory",
        )))
    }
}

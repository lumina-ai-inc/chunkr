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

pub fn convert_to_pdf(input_file: &NamedTempFile, mime_type: &str) -> Result<NamedTempFile, Box<dyn Error>> {
    let output_dir = input_file.path().parent().unwrap();

    let mut args = vec!["--headless"];
    // For images, use Draw with specific export options
    if mime_type.starts_with("image/") {
        args.extend(&[
            "--infilter=draw",
            "--convert-to",
            "pdf:draw_pdf_Export",  // Use Draw's PDF export
            "-env:UserInstallation=file:///tmp/libreoffice_convert",  // Prevent config conflicts
            "--writer",
            "-property:PageWidth:0",  // Remove page constraints
            "-property:PageHeight:0",
        ]);
    } else {
        args.extend(&[
            "--convert-to",
            "pdf",
        ]);
    }

    args.extend(&[
        "--outdir",
        output_dir.to_str().unwrap(),
        input_file.path().to_str().unwrap(),
    ]);

    let output = Command::new("libreoffice")
        .args(&args)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_convert_to_pdf() -> Result<(), Box<dyn Error>> {
        // Create a temporary directory for our test
        let output_dir = tempdir()?;
        
        // Create a temporary file from the input path
        let input_path = "./input/test.jpg"; // You'll replace this
        let temp_file = {
            let temp = NamedTempFile::new()?;
            fs::copy(input_path.clone(), temp.path())?;
            temp
        };

        // Perform the conversion
        let mime_type = check_file_type(&temp_file)?;
        let result = convert_to_pdf(&temp_file, &mime_type)?;
        
        // Copy the result to the output directory with original filename
        let original_name = PathBuf::from(input_path.clone())
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let output_path = output_dir.path().join(format!("{}.pdf", original_name));
        fs::copy(result.path(), &output_path)?;

        // Verify the output file exists and has content
        assert!(output_path.exists());
        assert!(fs::metadata(&output_path)?.len() > 0);

        Ok(())
    }
}